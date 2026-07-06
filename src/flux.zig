//! Native FLUX.2-klein-4B text→image (Qwen3 text encoder + parallel-attn DiT +
//! FlowMatchEuler sampler), ported from `mflux` (pure-MLX) to mlx-c FFI.
//! VAE decoder lives in flux_vae.zig. See docs/native-mediagen/02-image-flux.md.
//!
//! mflux pre-quantized weights are affine 4-bit (U32 weight + bf16 scales +
//! bf16 biases, group_size 64). We dequantize-free via mlx_quantized_matmul.
//! Convs (VAE only) are bf16 in MLX OHWI layout already.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const sse = @import("gen_sse.zig");
const lora_mod = @import("lora.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;

// ── Low-level helpers ──

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
inline fn rms(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, s));
    return o;
}
inline fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}
inline fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, s));
    return mulA(x, sig, s);
}
fn concat(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}
/// Slice along `axis` of a 3-D [d0,d1,d2] array: [start,stop) on that axis.
fn slice3(x: mlx.mlx_array, axis: usize, start: c_int, stop: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var lo = [_]c_int{ 0, 0, 0 };
    var hi = [_]c_int{ sh[0], sh[1], sh[2] };
    const st = [_]c_int{ 1, 1, 1 };
    lo[axis] = start;
    hi[axis] = stop;
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &lo, 3, &hi, 3, &st, 3, s));
    return o;
}

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
    lora: ?lora_mod.Ref = null, // runtime adapter (non-owning; gen.zig owns the File)

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
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.scales);
        _ = mlx.mlx_array_free(self.biases);
        if (self.add_bias) |b| _ = mlx.mlx_array_free(b);
    }
    fn forward(self: *const QLinear, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(&o, x, self.w, self.scales, self.biases, true, mlx.mlx_optional_int.some(@intCast(self.group_size)), mlx.mlx_optional_int.some(@intCast(self.bits)), "affine", s));
        if (self.add_bias) |b| {
            const r = try addA(o, b, s);
            _ = mlx.mlx_array_free(o);
            o = r;
        }
        if (self.lora) |lr| {
            const d = try lora_mod.delta(x, lr, s);
            defer _ = mlx.mlx_array_free(d);
            const r = try addA(o, d, s);
            _ = mlx.mlx_array_free(o);
            o = r;
        }
        return o;
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

    /// Encode token ids [1, seq] (int32) with attention_mask [1, seq] → capture
    /// raw hidden states at layers 9/18/27 → prompt_embeds [1, seq, 3*hidden].
    pub fn encode(self: *TextEncoder, ids: []const i32, mask: []const i32) !mlx.mlx_array {
        const s = self.s;
        const c = self.cfg;
        const seq: c_int = @intCast(ids.len);
        const H: u32 = c.te_hidden;

        // embed lookup
        const id_shape = [_]c_int{seq};
        const id_arr = mlx.mlx_array_new_data(ids.ptr, &id_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(id_arr);
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_axis(&taken, self.embed_table, id_arr, 0, s));
        var x = try reshape(taken, &[_]c_int{ 1, seq, @intCast(H) }, s);

        // Build additive mask [1,1,seq,seq] = causal + padding (in f32).
        const attn_mask = try self.buildMask(mask, seq);
        defer _ = mlx.mlx_array_free(attn_mask);

        // capture indices: list[0]=embed, list[k]=after layer k-1. Want 9,18,27.
        const want = [_]usize{ 9, 18, 27 };
        var caps: [3]?mlx.mlx_array = .{ null, null, null };
        errdefer for (caps) |cp| {
            if (cp) |a| _ = mlx.mlx_array_free(a);
        };

        for (self.layers, 0..) |*layer, li| {
            const nx = try self.layerForward(x, layer, attn_mask, seq, s);
            _ = mlx.mlx_array_free(x);
            x = nx;
            // list index after this layer is li+1
            const idx = li + 1;
            for (want, 0..) |wv, wi| {
                if (wv == idx) {
                    var cp = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_array_set(&cp, x));
                    caps[wi] = cp;
                }
            }
        }
        _ = mlx.mlx_array_free(x);

        // stack [1,3,seq,H] → transpose [1,seq,3,H] → reshape [1,seq,3H]
        const stacked = try concat(&[_]mlx.mlx_array{ caps[0].?, caps[1].?, caps[2].? }, 0, s); // [3, seq, H]
        for (caps) |cp| _ = mlx.mlx_array_free(cp.?);
        defer _ = mlx.mlx_array_free(stacked);
        // stacked is [3, 1, seq, H] (each cap was [1,seq,H]); concat axis0 → [3,seq,H]? caps are [1,seq,H] → concat axis0 → [3,seq,H]
        const st4 = try reshape(stacked, &[_]c_int{ 3, seq, @intCast(H) }, s);
        defer _ = mlx.mlx_array_free(st4);
        const tr = try transpose(st4, &[_]c_int{ 1, 0, 2 }, s); // [seq, 3, H]
        defer _ = mlx.mlx_array_free(tr);
        return reshape(tr, &[_]c_int{ 1, seq, @intCast(3 * H) }, s);
    }

    fn buildMask(self: *TextEncoder, mask: []const i32, seq: c_int) !mlx.mlx_array {
        const n: usize = @intCast(seq);
        const buf = try self.allocator.alloc(f32, n * n);
        defer self.allocator.free(buf);
        const neg = -std.math.inf(f32);
        for (0..n) |i| {
            for (0..n) |j| {
                // causal: j>i → -inf; padding: mask[j]==0 → -inf
                const causal_block = j > i;
                const pad_block = mask[j] == 0;
                buf[i * n + j] = if (causal_block or pad_block) neg else 0.0;
            }
        }
        const shape = [_]c_int{ 1, 1, seq, seq };
        const f = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
        return f;
    }

    fn layerForward(self: *TextEncoder, x: mlx.mlx_array, layer: *const TeLayer, mask: mlx.mlx_array, seq: c_int, s: S) !mlx.mlx_array {
        const c = self.cfg;
        const eps = c.te_rms_eps;
        const heads: c_int = @intCast(c.te_heads);
        const kv: c_int = @intCast(c.te_kv);
        const hd: c_int = @intCast(c.te_head_dim);

        const xn = try rms(x, layer.input_ln, eps, s);
        defer _ = mlx.mlx_array_free(xn);
        const q = try layer.q.forward(xn, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try layer.k.forward(xn, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try layer.v.forward(xn, s);
        defer _ = mlx.mlx_array_free(v);
        // reshape [1,seq,heads,hd], q/k norm over hd, transpose [1,heads,seq,hd]
        const q4 = try reshape(q, &[_]c_int{ 1, seq, heads, hd }, s);
        defer _ = mlx.mlx_array_free(q4);
        const qn = try rms(q4, layer.q_norm, eps, s);
        defer _ = mlx.mlx_array_free(qn);
        const qt = try transpose(qn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(qt);
        const k4 = try reshape(k, &[_]c_int{ 1, seq, kv, hd }, s);
        defer _ = mlx.mlx_array_free(k4);
        const kn = try rms(k4, layer.k_norm, eps, s);
        defer _ = mlx.mlx_array_free(kn);
        const kt = try transpose(kn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(kt);
        const v4 = try reshape(v, &[_]c_int{ 1, seq, kv, hd }, s);
        defer _ = mlx.mlx_array_free(v4);
        const vt = try transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(vt);
        // RoPE (NeoX) θ 1e6
        const base = mlx.mlx_optional_float.some(c.te_rope_theta);
        var qr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qr);
        try mlx.check(mlx.mlx_fast_rope(&qr, qt, hd, false, base, 1.0, 0, .{ .ctx = null }, s));
        var kr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kr);
        try mlx.check(mlx.mlx_fast_rope(&kr, kt, hd, false, base, 1.0, 0, .{ .ctx = null }, s));
        // SDPA in f32 with explicit mask
        const qf = try astype(qr, .float32, s);
        defer _ = mlx.mlx_array_free(qf);
        const kf = try astype(kr, .float32, s);
        defer _ = mlx.mlx_array_free(kf);
        const vf = try astype(vt, .float32, s);
        defer _ = mlx.mlx_array_free(vf);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(c.te_head_dim)));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_sink = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qf, kf, vf, scale, "array", mask, null_sink, s));
        const attn_bf = try astype(attn, .bfloat16, s);
        defer _ = mlx.mlx_array_free(attn_bf);
        const at = try transpose(attn_bf, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        const af = try reshape(at, &[_]c_int{ 1, seq, heads * hd }, s);
        defer _ = mlx.mlx_array_free(af);
        const o = try layer.o.forward(af, s);
        defer _ = mlx.mlx_array_free(o);
        const h1 = try addA(x, o, s);
        defer _ = mlx.mlx_array_free(h1);
        // MLP
        const hn = try rms(h1, layer.post_ln, eps, s);
        defer _ = mlx.mlx_array_free(hn);
        const g = try layer.gate.forward(hn, s);
        defer _ = mlx.mlx_array_free(g);
        const sg = try silu(g, s);
        defer _ = mlx.mlx_array_free(sg);
        const u = try layer.up.forward(hn, s);
        defer _ = mlx.mlx_array_free(u);
        const gu = try mulA(sg, u, s);
        defer _ = mlx.mlx_array_free(gu);
        const dn = try layer.down.forward(gu, s);
        defer _ = mlx.mlx_array_free(dn);
        return addA(h1, dn, s);
    }
};

pub fn loadTextEncoder(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !TextEncoder {
    const dir = try fmtKey(allocator, "{s}/text_encoder", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    var te: TextEncoder = undefined;
    te.cfg = .{};
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


// ════════════════════════════════════════════════════════════════════════
// Flux2 DiT (parallel-attn): 5 double + 20 single blocks, shared modulation.
// ════════════════════════════════════════════════════════════════════════

inline fn meanLast(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_mean_axis(&o, x, -1, true, s));
    return o;
}
/// LayerNorm with affine=false: (x-mean)/sqrt(var+eps) over last dim.
fn layerNormNA(x: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const m = try meanLast(xf, s);
    defer _ = mlx.mlx_array_free(m);
    const xc = try subA(xf, m, s);
    defer _ = mlx.mlx_array_free(xc);
    const sq = try mulA(xc, xc, s);
    defer _ = mlx.mlx_array_free(sq);
    const v = try meanLast(sq, s);
    defer _ = mlx.mlx_array_free(v);
    const epsa = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(epsa);
    var ve = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ve);
    try mlx.check(mlx.mlx_add(&ve, v, epsa, s));
    var rs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rs);
    try mlx.check(mlx.mlx_rsqrt(&rs, ve, s));
    const out = try mulA(xc, rs, s);
    defer _ = mlx.mlx_array_free(out);
    return astype(out, .bfloat16, s);
}

/// modulated = (1+scale)*ln + shift   (all [1,seq,D] / [1,1,D] broadcast)
fn modulate(ln: mlx.mlx_array, scale: mlx.mlx_array, shift: mlx.mlx_array, s: S) !mlx.mlx_array {
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var sp1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sp1);
    try mlx.check(mlx.mlx_add(&sp1, scale, one, s));
    const m = try mulA(ln, sp1, s);
    defer _ = mlx.mlx_array_free(m);
    return addA(m, shift, s);
}

const DoubleBlock = struct {
    q: QLinear, k: QLinear, v: QLinear, o: QLinear,
    add_q: QLinear, add_k: QLinear, add_v: QLinear, add_o: QLinear,
    nq: mlx.mlx_array, nk: mlx.mlx_array, naq: mlx.mlx_array, nak: mlx.mlx_array,
    ff_in: QLinear, ff_out: QLinear,
    ffc_in: QLinear, ffc_out: QLinear,
    fn deinit(self: *DoubleBlock) void {
        inline for (.{"q","k","v","o","add_q","add_k","add_v","add_o","ff_in","ff_out","ffc_in","ffc_out"}) |f| @field(self, f).deinit();
        inline for (.{"nq","nk","naq","nak"}) |f| _ = mlx.mlx_array_free(@field(self, f));
    }
};
const SingleBlock = struct {
    qkv_mlp: QLinear, o: QLinear, nq: mlx.mlx_array, nk: mlx.mlx_array,
    fn deinit(self: *SingleBlock) void {
        self.qkv_mlp.deinit(); self.o.deinit();
        _ = mlx.mlx_array_free(self.nq); _ = mlx.mlx_array_free(self.nk);
    }
};

pub const Dit = struct {
    cfg: FluxConfig,
    allocator: std.mem.Allocator,
    s: S,
    x_embedder: QLinear,
    context_embedder: QLinear,
    t_lin1: QLinear, t_lin2: QLinear,
    mod_img: QLinear, mod_txt: QLinear, mod_single: QLinear,
    doubles: []DoubleBlock,
    singles: []SingleBlock,
    norm_out_lin: QLinear,
    proj_out: QLinear,

    pub fn deinit(self: *Dit) void {
        self.x_embedder.deinit(); self.context_embedder.deinit();
        self.t_lin1.deinit(); self.t_lin2.deinit();
        self.mod_img.deinit(); self.mod_txt.deinit(); self.mod_single.deinit();
        for (self.doubles) |*b| b.deinit();
        self.allocator.free(self.doubles);
        for (self.singles) |*b| b.deinit();
        self.allocator.free(self.singles);
        self.norm_out_lin.deinit(); self.proj_out.deinit();
    }

    /// Build interleaved-RoPE cos/sin [seq,64] from ids [seq,4] (computed on CPU).
    fn buildRope(self: *Dit, ids: []const i32, seq: usize) !struct { cos: mlx.mlx_array, sin: mlx.mlx_array } {
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
        const cf = mlx.mlx_array_new_data(cosb.ptr, &shape, 2, .float32);
        const sf = mlx.mlx_array_new_data(sinb.ptr, &shape, 2, .float32);
        return .{ .cos = cf, .sin = sf };
    }

    /// apply interleaved RoPE to q [1,H,seq,128] using cos/sin [seq,64] (f32).
    fn applyRope(self: *Dit, q: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, seq: c_int, heads: c_int) !mlx.mlx_array {
        const s = self.s;
        const qf = try astype(q, .float32, s);
        defer _ = mlx.mlx_array_free(qf);
        const q5 = try reshape(qf, &[_]c_int{ 1, heads, seq, 64, 2 }, s); // [.,64,2]
        defer _ = mlx.mlx_array_free(q5);
        // real/imag via slice on last axis
        const real = try sliceLast2(q5, 0, s);
        defer _ = mlx.mlx_array_free(real);
        const imag = try sliceLast2(q5, 1, s);
        defer _ = mlx.mlx_array_free(imag);
        const cos_b = try reshape(cos, &[_]c_int{ 1, 1, seq, 64 }, s);
        defer _ = mlx.mlx_array_free(cos_b);
        const sin_b = try reshape(sin, &[_]c_int{ 1, 1, seq, 64 }, s);
        defer _ = mlx.mlx_array_free(sin_b);
        // out0 = real*cos - imag*sin ; out1 = imag*cos + real*sin
        const rc = try mulA(real, cos_b, s); defer _ = mlx.mlx_array_free(rc);
        const is_ = try mulA(imag, sin_b, s); defer _ = mlx.mlx_array_free(is_);
        const out0 = try subA(rc, is_, s); defer _ = mlx.mlx_array_free(out0);
        const ic = try mulA(imag, cos_b, s); defer _ = mlx.mlx_array_free(ic);
        const rs2 = try mulA(real, sin_b, s); defer _ = mlx.mlx_array_free(rs2);
        const out1 = try addA(ic, rs2, s); defer _ = mlx.mlx_array_free(out1);
        // stack last axis: [.,64,1] each then concat → [.,64,2] → reshape [.,128]
        const o0e = try reshape(out0, &[_]c_int{ 1, heads, seq, 64, 1 }, s); defer _ = mlx.mlx_array_free(o0e);
        const o1e = try reshape(out1, &[_]c_int{ 1, heads, seq, 64, 1 }, s); defer _ = mlx.mlx_array_free(o1e);
        const st = try concat(&[_]mlx.mlx_array{ o0e, o1e }, 4, s); defer _ = mlx.mlx_array_free(st);
        const flat = try reshape(st, &[_]c_int{ 1, heads, seq, 128 }, s); defer _ = mlx.mlx_array_free(flat);
        return astype(flat, .bfloat16, s);
    }

    /// Full DiT forward. latents [1,Ni,128], enc [1,Nt,7680], temb-driving timestep
    /// scalar, img_ids/txt_ids slices. Returns noise [1,Ni,128].
    pub fn forward(self: *Dit, latents: mlx.mlx_array, enc: mlx.mlx_array, timestep: f32, img_ids: []const i32, txt_ids: []const i32) !mlx.mlx_array {
        const s = self.s;
        const c = self.cfg;
        const inner: c_int = @intCast(c.inner_dim);
        const heads: c_int = @intCast(c.heads);
        const hd: c_int = @intCast(c.head_dim);
        const Nt: c_int = mlx.getShape(enc)[1];
        const Ni: c_int = mlx.getShape(latents)[1];
        const Nj = Nt + Ni;

        // temb = time_guidance_embed(timestep)
        const temb = try self.timeEmbed(timestep);
        defer _ = mlx.mlx_array_free(temb);

        // embed
        var hs = try self.x_embedder.forward(latents, s); // [1,Ni,inner]
        var ehs = try self.context_embedder.forward(enc, s); // [1,Nt,inner]

        // rope for txt then img, concat
        const tr = try self.buildRope(txt_ids, @intCast(Nt));
        defer { _ = mlx.mlx_array_free(tr.cos); _ = mlx.mlx_array_free(tr.sin); }
        const ir = try self.buildRope(img_ids, @intCast(Ni));
        defer { _ = mlx.mlx_array_free(ir.cos); _ = mlx.mlx_array_free(ir.sin); }
        const cos = try concat(&[_]mlx.mlx_array{ tr.cos, ir.cos }, 0, s); defer _ = mlx.mlx_array_free(cos);
        const sin = try concat(&[_]mlx.mlx_array{ tr.sin, ir.sin }, 0, s); defer _ = mlx.mlx_array_free(sin);

        // modulation params (Flux2Modulation applies silu(temb) before the linear)
        const stemb = try silu(temb, s); defer _ = mlx.mlx_array_free(stemb);
        const mi = try self.mod_img.forward(stemb, s); defer _ = mlx.mlx_array_free(mi); // [1,1,6*inner]
        const mt = try self.mod_txt.forward(stemb, s); defer _ = mlx.mlx_array_free(mt);
        const ms = try self.mod_single.forward(stemb, s); defer _ = mlx.mlx_array_free(ms); // [1,1,3*inner]

        for (self.doubles) |*b| {
            const r = try self.doubleBlock(hs, ehs, b, mi, mt, cos, sin, Nt, Ni, heads, hd, inner, s);
            _ = mlx.mlx_array_free(hs); _ = mlx.mlx_array_free(ehs);
            hs = r.img; ehs = r.txt;
        }
        // concat [text, image]
        var joint = try concat(&[_]mlx.mlx_array{ ehs, hs }, 1, s);
        _ = mlx.mlx_array_free(hs); _ = mlx.mlx_array_free(ehs);
        for (self.singles) |*b| {
            const nj = try self.singleBlock(joint, b, ms, cos, sin, Nj, heads, hd, inner, s);
            _ = mlx.mlx_array_free(joint); joint = nj;
        }
        // slice off text
        const img = try slice3(joint, 1, Nt, Nj, s);
        _ = mlx.mlx_array_free(joint);
        defer _ = mlx.mlx_array_free(img);
        // norm_out (AdaLayerNormContinuous): linear(silu(temb)) → [scale,shift]
        const stmb = try silu(temb, s); defer _ = mlx.mlx_array_free(stmb);
        const so = try self.norm_out_lin.forward(stmb, s); defer _ = mlx.mlx_array_free(so); // [1,1,2*inner]
        const scale = try slice3(so, 2, 0, inner, s); defer _ = mlx.mlx_array_free(scale);
        const shift = try slice3(so, 2, inner, 2 * inner, s); defer _ = mlx.mlx_array_free(shift);
        const ln = try layerNormNA(img, 1e-6, s); defer _ = mlx.mlx_array_free(ln);
        const modded = try modulate(ln, scale, shift, s); defer _ = mlx.mlx_array_free(modded);
        return self.proj_out.forward(modded, s); // [1,Ni,128]
    }

    fn timeEmbed(self: *Dit, t: f32) !mlx.mlx_array {
        const s = self.s;
        // sinusoidal(256), flip_sin_to_cos
        var buf: [256]f32 = undefined;
        const half = 128;
        for (0..half) |i| {
            const freq = std.math.exp(-std.math.log(f32, std.math.e, 10000.0) * @as(f32, @floatFromInt(i)) / @as(f32, half));
            const arg = t * freq;
            buf[i] = @sin(arg);
            buf[half + i] = @cos(arg);
        }
        // flip: cat(emb[half:], emb[:half]) = [cos..., sin...]
        var flipped: [256]f32 = undefined;
        @memcpy(flipped[0..half], buf[half..256]);
        @memcpy(flipped[half..256], buf[0..half]);
        const shape = [_]c_int{ 1, 256 };
        const te = mlx.mlx_array_new_data(&flipped, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(te);
        const te_bf = try astype(te, .bfloat16, s);
        defer _ = mlx.mlx_array_free(te_bf);
        const l1 = try self.t_lin1.forward(te_bf, s); defer _ = mlx.mlx_array_free(l1);
        const a = try silu(l1, s); defer _ = mlx.mlx_array_free(a);
        const t2 = try self.t_lin2.forward(a, s); defer _ = mlx.mlx_array_free(t2); // [1, inner]
        return reshape(t2, &[_]c_int{ 1, 1, @intCast(self.cfg.inner_dim) }, s); // [1,1,inner]
    }

    fn attention(self: *Dit, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, seq: c_int, heads: c_int, hd: c_int, s: S) !mlx.mlx_array {
        _ = self;
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(hd)));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_a = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, q, k, v, scale, "", null_a, null_a, s));
        const at = try transpose(attn, &[_]c_int{ 0, 2, 1, 3 }, s); defer _ = mlx.mlx_array_free(at);
        return reshape(at, &[_]c_int{ 1, seq, heads * hd }, s);
    }

    fn splitHeads(self: *Dit, x: mlx.mlx_array, seq: c_int, heads: c_int, hd: c_int, norm: mlx.mlx_array, s: S) !mlx.mlx_array {
        _ = self;
        const x4 = try reshape(x, &[_]c_int{ 1, seq, heads, hd }, s); defer _ = mlx.mlx_array_free(x4);
        const xt = try transpose(x4, &[_]c_int{ 0, 2, 1, 3 }, s); defer _ = mlx.mlx_array_free(xt);
        // q/k norm over hd in f32
        const xf = try astype(xt, .float32, s); defer _ = mlx.mlx_array_free(xf);
        const xn = try rms(xf, norm, 1e-5, s); defer _ = mlx.mlx_array_free(xn);
        return astype(xn, .bfloat16, s);
    }

    fn doubleBlock(self: *Dit, hs: mlx.mlx_array, ehs: mlx.mlx_array, b: *const DoubleBlock, mi: mlx.mlx_array, mt: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, Nt: c_int, Ni: c_int, heads: c_int, hd: c_int, inner: c_int, s: S) !struct { img: mlx.mlx_array, txt: mlx.mlx_array } {
        // unpack mod: img has 6 chunks [shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp]
        const im = try modChunks(mi, inner, 6, s); defer for (im) |x| { _ = mlx.mlx_array_free(x); };
        const tm = try modChunks(mt, inner, 6, s); defer for (tm) |x| { _ = mlx.mlx_array_free(x); };
        // norm + modulate
        const nh = try layerNormNA(hs, 1e-6, s); defer _ = mlx.mlx_array_free(nh);
        const nhm = try modulate(nh, im[1], im[0], s); defer _ = mlx.mlx_array_free(nhm);
        const ne = try layerNormNA(ehs, 1e-6, s); defer _ = mlx.mlx_array_free(ne);
        const nem = try modulate(ne, tm[1], tm[0], s); defer _ = mlx.mlx_array_free(nem);
        // qkv for image
        const q = try b.q.forward(nhm, s); defer _ = mlx.mlx_array_free(q);
        const k = try b.k.forward(nhm, s); defer _ = mlx.mlx_array_free(k);
        const v = try b.v.forward(nhm, s); defer _ = mlx.mlx_array_free(v);
        const qh = try self.splitHeads(q, Ni, heads, hd, b.nq, s); defer _ = mlx.mlx_array_free(qh);
        const kh = try self.splitHeads(k, Ni, heads, hd, b.nk, s); defer _ = mlx.mlx_array_free(kh);
        const vh4 = try reshape(v, &[_]c_int{ 1, Ni, heads, hd }, s); defer _ = mlx.mlx_array_free(vh4);
        const vh = try transpose(vh4, &[_]c_int{ 0, 2, 1, 3 }, s); defer _ = mlx.mlx_array_free(vh);
        // qkv for text
        const eq = try b.add_q.forward(nem, s); defer _ = mlx.mlx_array_free(eq);
        const ek = try b.add_k.forward(nem, s); defer _ = mlx.mlx_array_free(ek);
        const ev = try b.add_v.forward(nem, s); defer _ = mlx.mlx_array_free(ev);
        const eqh = try self.splitHeads(eq, Nt, heads, hd, b.naq, s); defer _ = mlx.mlx_array_free(eqh);
        const ekh = try self.splitHeads(ek, Nt, heads, hd, b.nak, s); defer _ = mlx.mlx_array_free(ekh);
        const evh4 = try reshape(ev, &[_]c_int{ 1, Nt, heads, hd }, s); defer _ = mlx.mlx_array_free(evh4);
        const evh = try transpose(evh4, &[_]c_int{ 0, 2, 1, 3 }, s); defer _ = mlx.mlx_array_free(evh);
        // concat [text, image] on seq axis (axis 2)
        const qj = try concat(&[_]mlx.mlx_array{ eqh, qh }, 2, s); defer _ = mlx.mlx_array_free(qj);
        const kj = try concat(&[_]mlx.mlx_array{ ekh, kh }, 2, s); defer _ = mlx.mlx_array_free(kj);
        const vj = try concat(&[_]mlx.mlx_array{ evh, vh }, 2, s); defer _ = mlx.mlx_array_free(vj);
        const Nj = Nt + Ni;
        const qr = try self.applyRope(qj, cos, sin, Nj, heads); defer _ = mlx.mlx_array_free(qr);
        const kr = try self.applyRope(kj, cos, sin, Nj, heads); defer _ = mlx.mlx_array_free(kr);
        const attn = try self.attention(qr, kr, vj, Nj, heads, hd, s); defer _ = mlx.mlx_array_free(attn);
        // split back
        const a_txt = try slice3(attn, 1, 0, Nt, s); defer _ = mlx.mlx_array_free(a_txt);
        const a_img = try slice3(attn, 1, Nt, Nj, s); defer _ = mlx.mlx_array_free(a_img);
        const ao_img = try b.o.forward(a_img, s); defer _ = mlx.mlx_array_free(ao_img);
        const ao_txt = try b.add_o.forward(a_txt, s); defer _ = mlx.mlx_array_free(ao_txt);
        // gated residual
        const g_img = try mulA(im[2], ao_img, s); defer _ = mlx.mlx_array_free(g_img);
        const hs1 = try addA(hs, g_img, s); defer _ = mlx.mlx_array_free(hs1);
        const g_txt = try mulA(tm[2], ao_txt, s); defer _ = mlx.mlx_array_free(g_txt);
        const ehs1 = try addA(ehs, g_txt, s); defer _ = mlx.mlx_array_free(ehs1);
        // FF image
        const nh2 = try layerNormNA(hs1, 1e-6, s); defer _ = mlx.mlx_array_free(nh2);
        const nh2m = try modulate(nh2, im[4], im[3], s); defer _ = mlx.mlx_array_free(nh2m);
        const ff_img = try self.feedForward(nh2m, &b.ff_in, &b.ff_out, s); defer _ = mlx.mlx_array_free(ff_img);
        const gff_img = try mulA(im[5], ff_img, s); defer _ = mlx.mlx_array_free(gff_img);
        const img_out = try addA(hs1, gff_img, s);
        // FF text
        const ne2 = try layerNormNA(ehs1, 1e-6, s); defer _ = mlx.mlx_array_free(ne2);
        const ne2m = try modulate(ne2, tm[4], tm[3], s); defer _ = mlx.mlx_array_free(ne2m);
        const ff_txt = try self.feedForward(ne2m, &b.ffc_in, &b.ffc_out, s); defer _ = mlx.mlx_array_free(ff_txt);
        const gff_txt = try mulA(tm[5], ff_txt, s); defer _ = mlx.mlx_array_free(gff_txt);
        const txt_out = try addA(ehs1, gff_txt, s);
        return .{ .img = img_out, .txt = txt_out };
    }

    fn singleBlock(self: *Dit, hs: mlx.mlx_array, b: *const SingleBlock, ms: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, Nj: c_int, heads: c_int, hd: c_int, inner: c_int, s: S) !mlx.mlx_array {
        const m = try modChunks(ms, inner, 3, s); defer for (m) |x| { _ = mlx.mlx_array_free(x); };
        const nh = try layerNormNA(hs, 1e-6, s); defer _ = mlx.mlx_array_free(nh);
        const nhm = try modulate(nh, m[1], m[0], s); defer _ = mlx.mlx_array_free(nhm);
        const proj = try b.qkv_mlp.forward(nhm, s); defer _ = mlx.mlx_array_free(proj); // [1,Nj, 3*inner + 2*mlp]
        const mlp_h: c_int = @intFromFloat(@as(f32, @floatFromInt(inner)) * self.cfg.mlp_ratio);
        const qkv = try slice3(proj, 2, 0, 3 * inner, s); defer _ = mlx.mlx_array_free(qkv);
        const mlp = try slice3(proj, 2, 3 * inner, 3 * inner + 2 * mlp_h, s); defer _ = mlx.mlx_array_free(mlp);
        const q = try slice3(qkv, 2, 0, inner, s); defer _ = mlx.mlx_array_free(q);
        const k = try slice3(qkv, 2, inner, 2 * inner, s); defer _ = mlx.mlx_array_free(k);
        const v = try slice3(qkv, 2, 2 * inner, 3 * inner, s); defer _ = mlx.mlx_array_free(v);
        const qh = try self.splitHeads(q, Nj, heads, hd, b.nq, s); defer _ = mlx.mlx_array_free(qh);
        const kh = try self.splitHeads(k, Nj, heads, hd, b.nk, s); defer _ = mlx.mlx_array_free(kh);
        const vh4 = try reshape(v, &[_]c_int{ 1, Nj, heads, hd }, s); defer _ = mlx.mlx_array_free(vh4);
        const vh = try transpose(vh4, &[_]c_int{ 0, 2, 1, 3 }, s); defer _ = mlx.mlx_array_free(vh);
        const qr = try self.applyRope(qh, cos, sin, Nj, heads); defer _ = mlx.mlx_array_free(qr);
        const kr = try self.applyRope(kh, cos, sin, Nj, heads); defer _ = mlx.mlx_array_free(kr);
        const attn = try self.attention(qr, kr, vh, Nj, heads, hd, s); defer _ = mlx.mlx_array_free(attn);
        // SwiGLU on mlp: split 2 → silu(x1)*x2
        const x1 = try slice3(mlp, 2, 0, mlp_h, s); defer _ = mlx.mlx_array_free(x1);
        const x2 = try slice3(mlp, 2, mlp_h, 2 * mlp_h, s); defer _ = mlx.mlx_array_free(x2);
        const sx1 = try silu(x1, s); defer _ = mlx.mlx_array_free(sx1);
        const mlp_act = try mulA(sx1, x2, s); defer _ = mlx.mlx_array_free(mlp_act);
        const cat = try concat(&[_]mlx.mlx_array{ attn, mlp_act }, 2, s); defer _ = mlx.mlx_array_free(cat);
        const ao = try b.o.forward(cat, s); defer _ = mlx.mlx_array_free(ao);
        const g = try mulA(m[2], ao, s); defer _ = mlx.mlx_array_free(g);
        return addA(hs, g, s);
    }

    fn feedForward(self: *Dit, x: mlx.mlx_array, lin_in: *const QLinear, lin_out: *const QLinear, s: S) !mlx.mlx_array {
        _ = self;
        const proj = try lin_in.forward(x, s); defer _ = mlx.mlx_array_free(proj); // [.,2*inner_ff]
        const sh = mlx.getShape(proj);
        const half = @divExact(sh[2], 2);
        const x1 = try slice3(proj, 2, 0, half, s); defer _ = mlx.mlx_array_free(x1);
        const x2 = try slice3(proj, 2, half, 2 * half, s); defer _ = mlx.mlx_array_free(x2);
        const sx1 = try silu(x1, s); defer _ = mlx.mlx_array_free(sx1);
        const act = try mulA(sx1, x2, s); defer _ = mlx.mlx_array_free(act);
        return lin_out.forward(act, s);
    }
};

/// Split a modulation output [1,1,n*inner] into n chunks of [1,1,inner].
fn modChunks(mod: mlx.mlx_array, inner: c_int, n: usize, s: S) ![6]mlx.mlx_array {
    var out: [6]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null }, .{ .ctx = null }, .{ .ctx = null }, .{ .ctx = null }, .{ .ctx = null } };
    for (0..n) |i| {
        out[i] = try slice3(mod, 2, @intCast(@as(c_int, @intCast(i)) * inner), @intCast(@as(c_int, @intCast(i + 1)) * inner), s);
    }
    return out;
}

/// Slice index `idx` (0 or 1) of the last axis of a [..,64,2] array → [..,64].
fn sliceLast2(x: mlx.mlx_array, idx: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,H,seq,64,2]
    var lo = [_]c_int{ 0, 0, 0, 0, idx };
    var hi = [_]c_int{ sh[0], sh[1], sh[2], sh[3], idx + 1 };
    const st = [_]c_int{ 1, 1, 1, 1, 1 };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &lo, 5, &hi, 5, &st, 5, s));
    const sq = try reshape(o, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, s);
    _ = mlx.mlx_array_free(o);
    return sq;
}

pub fn loadDit(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !Dit {
    const dir = try fmtKey(allocator, "{s}/transformer", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    var d: Dit = undefined;
    d.cfg = .{}; d.allocator = allocator; d.s = s;
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
        const pfx = try fmtKey(allocator, "transformer_blocks.{d}", .{i}); defer allocator.free(pfx);
        b.* = try loadDouble(&w, allocator, pfx);
    }
    d.singles = try allocator.alloc(SingleBlock, d.cfg.single_layers);
    for (d.singles, 0..) |*b, i| {
        const pfx = try fmtKey(allocator, "single_transformer_blocks.{d}", .{i}); defer allocator.free(pfx);
        const qp = try fmtKey(allocator, "{s}.attn.to_qkv_mlp_proj", .{pfx}); defer allocator.free(qp);
        const op = try fmtKey(allocator, "{s}.attn.to_out", .{pfx}); defer allocator.free(op);
        const nq = try fmtKey(allocator, "{s}.attn.norm_q.weight", .{pfx}); defer allocator.free(nq);
        const nk = try fmtKey(allocator, "{s}.attn.norm_k.weight", .{pfx}); defer allocator.free(nk);
        b.* = .{ .qkv_mlp = try QLinear.load(&w, allocator, qp), .o = try QLinear.load(&w, allocator, op), .nq = try ownWeight(&w, nq), .nk = try ownWeight(&w, nk) };
    }
    return d;
}

/// Attach runtime LoRA adapters from `lf` to every matching DiT linear.
/// Non-owning: `lf` must outlive the attach. Returns the match count.
pub fn attachLora(dit: *Dit, lf: *const lora_mod.File, user_scale: f32) u32 {
    detachLora(dit);
    var matched: u32 = 0;
    var kbuf: [128]u8 = undefined;
    for (dit.doubles, 0..) |*b, i| {
        const mods = .{
            .{ "attn.to_q", &b.q },           .{ "attn.to_k", &b.k },
            .{ "attn.to_v", &b.v },           .{ "attn.to_out", &b.o },
            .{ "attn.add_q_proj", &b.add_q }, .{ "attn.add_k_proj", &b.add_k },
            .{ "attn.add_v_proj", &b.add_v }, .{ "attn.to_add_out", &b.add_o },
            .{ "ff.linear_in", &b.ff_in },    .{ "ff.linear_out", &b.ff_out },
            .{ "ff_context.linear_in", &b.ffc_in }, .{ "ff_context.linear_out", &b.ffc_out },
        };
        inline for (mods) |m| {
            const key = std.fmt.bufPrint(&kbuf, "transformer_blocks.{d}.{s}", .{ i, m[0] }) catch "";
            if (lf.find(key)) |e| {
                m[1].lora = .{ .at = e.at, .bt = e.bt, .scale = e.scale * user_scale };
                matched += 1;
            }
        }
    }
    for (dit.singles, 0..) |*b, i| {
        const mods = .{ .{ "attn.to_qkv_mlp_proj", &b.qkv_mlp }, .{ "attn.to_out", &b.o } };
        inline for (mods) |m| {
            const key = std.fmt.bufPrint(&kbuf, "single_transformer_blocks.{d}.{s}", .{ i, m[0] }) catch "";
            if (lf.find(key)) |e| {
                m[1].lora = .{ .at = e.at, .bt = e.bt, .scale = e.scale * user_scale };
                matched += 1;
            }
        }
    }
    return matched;
}

pub fn detachLora(dit: *Dit) void {
    for (dit.doubles) |*b| {
        inline for (.{ &b.q, &b.k, &b.v, &b.o, &b.add_q, &b.add_k, &b.add_v, &b.add_o, &b.ff_in, &b.ff_out, &b.ffc_in, &b.ffc_out }) |ql| ql.lora = null;
    }
    for (dit.singles) |*b| {
        b.qkv_mlp.lora = null;
        b.o.lora = null;
    }
}

fn loadDouble(w: *const Weights, a: std.mem.Allocator, pfx: []const u8) !DoubleBlock {
    const ld = struct {
        fn q(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8) !QLinear {
            const k = try fmtKey(aa, "{s}.{s}", .{ p, sub }); defer aa.free(k);
            return QLinear.load(ww, aa, k);
        }
        fn n(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8) !mlx.mlx_array {
            const k = try fmtKey(aa, "{s}.{s}", .{ p, sub }); defer aa.free(k);
            return ownWeight(ww, k);
        }
    };
    return .{
        .q = try ld.q(w, a, pfx, "attn.to_q"), .k = try ld.q(w, a, pfx, "attn.to_k"), .v = try ld.q(w, a, pfx, "attn.to_v"), .o = try ld.q(w, a, pfx, "attn.to_out"),
        .add_q = try ld.q(w, a, pfx, "attn.add_q_proj"), .add_k = try ld.q(w, a, pfx, "attn.add_k_proj"), .add_v = try ld.q(w, a, pfx, "attn.add_v_proj"), .add_o = try ld.q(w, a, pfx, "attn.to_add_out"),
        .nq = try ld.n(w, a, pfx, "attn.norm_q.weight"), .nk = try ld.n(w, a, pfx, "attn.norm_k.weight"), .naq = try ld.n(w, a, pfx, "attn.norm_added_q.weight"), .nak = try ld.n(w, a, pfx, "attn.norm_added_k.weight"),
        .ff_in = try ld.q(w, a, pfx, "ff.linear_in"), .ff_out = try ld.q(w, a, pfx, "ff.linear_out"),
        .ffc_in = try ld.q(w, a, pfx, "ff_context.linear_in"), .ffc_out = try ld.q(w, a, pfx, "ff_context.linear_out"),
    };
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
    const lat = try readF32(io, a, lat_p); defer a.free(lat);
    const pe = try readF32(io, a, pe_p); defer a.free(pe);
    const ii = try readI32(io, a, ii_p); defer a.free(ii);
    const ti = try readI32(io, a, ti_p); defer a.free(ti);
    const ref = try readF32(io, a, n0_p); defer a.free(ref);

    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var dit = try loadDit(io, a, s, model_dir); defer dit.deinit();

    // upload latents [1,4096,128] and pe [1,512,7680] as bf16
    const lat_sh = [_]c_int{ 1, 4096, 128 };
    const lat_f = mlx.mlx_array_new_data(lat.ptr, &lat_sh, 3, .float32); defer _ = mlx.mlx_array_free(lat_f);
    const lat_bf = try astype(lat_f, .bfloat16, s); defer _ = mlx.mlx_array_free(lat_bf);
    const pe_sh = [_]c_int{ 1, 512, 7680 };
    const pe_f = mlx.mlx_array_new_data(pe.ptr, &pe_sh, 3, .float32); defer _ = mlx.mlx_array_free(pe_f);
    const pe_bf = try astype(pe_f, .bfloat16, s); defer _ = mlx.mlx_array_free(pe_bf);

    const noise = try dit.forward(lat_bf, pe_bf, 1000.0, ii, ti);
    defer _ = mlx.mlx_array_free(noise);
    const nf = try astype(noise, .float32, s); defer _ = mlx.mlx_array_free(nf);
    _ = mlx.mlx_array_eval(nf);
    const n: usize = @intCast(mlx.mlx_array_size(nf));
    const data = mlx.mlx_array_data_float32(nf) orelse return error.NoData;
    try testing.expectEqual(ref.len, n);
    var dot: f64 = 0; var na: f64 = 0; var nb: f64 = 0;
    for (0..n) |i| { dot += @as(f64, data[i]) * ref[i]; na += @as(f64, data[i]) * data[i]; nb += @as(f64, ref[i]) * ref[i]; }
    const corr = dot / (std.math.sqrt(na) * std.math.sqrt(nb));
    std.debug.print("[flux-dit] n={d} corr={d:.6}\n", .{ n, corr });
    try testing.expect(corr > 0.99);
}


// ════════════════════════════════════════════════════════════════════════
// Flux2 VAE decoder: packed latents [1,128,64,64] → image [1,3,1024,1024].
// Convs are bf16 OHWI; data flows NHWC inside each conv/groupnorm.
// ════════════════════════════════════════════════════════════════════════

/// conv2d on NHWC input with bf16 OHWI weight [out,kh,kw,in] + bias [out].
fn conv2d(x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, pad: c_int, s: S) !mlx.mlx_array {
    // Materialize: mlx_conv2d silently miscomputes on strided/lazy-view inputs.
    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    try mlx.check(mlx.mlx_contiguous(&xc, x, false, s));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, 1, 1, pad, pad, 1, 1, 1, s));
    const r = try addA(o, bias, s);
    _ = mlx.mlx_array_free(o);
    return r;
}

/// PyTorch-compatible GroupNorm (32 groups) on NHWC [1,H,W,C], fp32, + affine.
fn groupNorm(x: mlx.mlx_array, weight: mlx.mlx_array, bias: mlx.mlx_array, groups: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,H,W,C]
    const H = sh[1];
    const Wd = sh[2];
    const C = sh[3];
    const cg = @divExact(C, groups);
    const xf = try astype(x, .float32, s); defer _ = mlx.mlx_array_free(xf);
    // [1,H,W,C] -> [1, H*W, groups, cg] -> [1, groups, H*W, cg] -> [1, groups, H*W*cg]
    const r1 = try reshape(xf, &[_]c_int{ 1, H * Wd, groups, cg }, s); defer _ = mlx.mlx_array_free(r1);
    const t1 = try transpose(r1, &[_]c_int{ 0, 2, 1, 3 }, s); defer _ = mlx.mlx_array_free(t1);
    const flat = try reshape(t1, &[_]c_int{ 1, groups, H * Wd * cg }, s); defer _ = mlx.mlx_array_free(flat);
    var mean = mlx.mlx_array_new(); defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, flat, -1, true, s));
    const xc = try subA(flat, mean, s); defer _ = mlx.mlx_array_free(xc);
    const sq = try mulA(xc, xc, s); defer _ = mlx.mlx_array_free(sq);
    var v = mlx.mlx_array_new(); defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_mean_axis(&v, sq, -1, true, s));
    const epsa = mlx.mlx_array_new_float(1e-6); defer _ = mlx.mlx_array_free(epsa);
    var ve = mlx.mlx_array_new(); defer _ = mlx.mlx_array_free(ve);
    try mlx.check(mlx.mlx_add(&ve, v, epsa, s));
    var rsq = mlx.mlx_array_new(); defer _ = mlx.mlx_array_free(rsq);
    try mlx.check(mlx.mlx_rsqrt(&rsq, ve, s));
    const norm = try mulA(xc, rsq, s); defer _ = mlx.mlx_array_free(norm);
    // back to NHWC
    const b1 = try reshape(norm, &[_]c_int{ 1, groups, H * Wd, cg }, s); defer _ = mlx.mlx_array_free(b1);
    const b2 = try transpose(b1, &[_]c_int{ 0, 2, 1, 3 }, s); defer _ = mlx.mlx_array_free(b2);
    const b3 = try reshape(b2, &[_]c_int{ 1, H, Wd, C }, s); defer _ = mlx.mlx_array_free(b3);
    // affine (weight/bias per channel, bf16)
    const wf = try astype(weight, .float32, s); defer _ = mlx.mlx_array_free(wf);
    const bf = try astype(bias, .float32, s); defer _ = mlx.mlx_array_free(bf);
    const sc = try mulA(b3, wf, s); defer _ = mlx.mlx_array_free(sc);
    const out = try addA(sc, bf, s); defer _ = mlx.mlx_array_free(out);
    return astype(out, .bfloat16, s);
}

const Resnet = struct {
    n1w: mlx.mlx_array, n1b: mlx.mlx_array, c1w: mlx.mlx_array, c1b: mlx.mlx_array,
    n2w: mlx.mlx_array, n2b: mlx.mlx_array, c2w: mlx.mlx_array, c2b: mlx.mlx_array,
    sw: ?mlx.mlx_array = null, sb: ?mlx.mlx_array = null, // conv_shortcut (1x1)
    fn deinit(self: *Resnet) void {
        inline for (.{ "n1w","n1b","c1w","c1b","n2w","n2b","c2w","c2b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        if (self.sw) |x| _ = mlx.mlx_array_free(x);
        if (self.sb) |x| _ = mlx.mlx_array_free(x);
    }
    fn forward(self: *const Resnet, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const h0 = try groupNorm(x, self.n1w, self.n1b, 32, s); defer _ = mlx.mlx_array_free(h0);
        const a0 = try silu(h0, s); defer _ = mlx.mlx_array_free(a0);
        const c1 = try conv2d(a0, self.c1w, self.c1b, 1, s); defer _ = mlx.mlx_array_free(c1);
        const h1 = try groupNorm(c1, self.n2w, self.n2b, 32, s); defer _ = mlx.mlx_array_free(h1);
        const a1 = try silu(h1, s); defer _ = mlx.mlx_array_free(a1);
        const c2 = try conv2d(a1, self.c2w, self.c2b, 1, s); defer _ = mlx.mlx_array_free(c2);
        if (self.sw) |sw| {
            const sc = try conv2d(x, sw, self.sb.?, 0, s); defer _ = mlx.mlx_array_free(sc);
            return addA(c2, sc, s);
        }
        return addA(c2, x, s);
    }
};

const VaeAttn = struct {
    gnw: mlx.mlx_array, gnb: mlx.mlx_array,
    q: QLinear, k: QLinear, v: QLinear, o: QLinear,
    fn deinit(self: *VaeAttn) void {
        _ = mlx.mlx_array_free(self.gnw); _ = mlx.mlx_array_free(self.gnb);
        self.q.deinit(); self.k.deinit(); self.v.deinit(); self.o.deinit();
    }
    fn forward(self: *const VaeAttn, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const sh = mlx.getShape(x); const H = sh[1]; const Wd = sh[2]; const C = sh[3];
        const normed = try groupNorm(x, self.gnw, self.gnb, 32, s); defer _ = mlx.mlx_array_free(normed);
        const q = try self.q.forward(normed, s); defer _ = mlx.mlx_array_free(q);
        const k = try self.k.forward(normed, s); defer _ = mlx.mlx_array_free(k);
        const v = try self.v.forward(normed, s); defer _ = mlx.mlx_array_free(v);
        // [1,H,W,C] -> [1, 1, H*W, C] (single head)
        const qr = try reshape(q, &[_]c_int{ 1, 1, H * Wd, C }, s); defer _ = mlx.mlx_array_free(qr);
        const kr = try reshape(k, &[_]c_int{ 1, 1, H * Wd, C }, s); defer _ = mlx.mlx_array_free(kr);
        const vr = try reshape(v, &[_]c_int{ 1, 1, H * Wd, C }, s); defer _ = mlx.mlx_array_free(vr);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(C)));
        var attn = mlx.mlx_array_new(); defer _ = mlx.mlx_array_free(attn);
        const null_a = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qr, kr, vr, scale, "", null_a, null_a, s));
        const ar = try reshape(attn, &[_]c_int{ 1, H, Wd, C }, s); defer _ = mlx.mlx_array_free(ar);
        const ao = try self.o.forward(ar, s); defer _ = mlx.mlx_array_free(ao);
        return addA(x, ao, s);
    }
};

pub const Vae = struct {
    allocator: std.mem.Allocator,
    s: S,
    bn_mean: mlx.mlx_array, bn_var: mlx.mlx_array,
    pq_w: mlx.mlx_array, pq_b: mlx.mlx_array, // post_quant_conv 1x1
    conv_in_w: mlx.mlx_array, conv_in_b: mlx.mlx_array,
    mid_r0: Resnet, mid_attn: VaeAttn, mid_r1: Resnet,
    up_resnets: [4][3]Resnet,
    up_conv_w: [3]mlx.mlx_array, up_conv_b: [3]mlx.mlx_array, // upsamplers for blocks 0,1,2
    norm_out_w: mlx.mlx_array, norm_out_b: mlx.mlx_array,
    conv_out_w: mlx.mlx_array, conv_out_b: mlx.mlx_array,

    pub fn deinit(self: *Vae) void {
        inline for (.{ "bn_mean","bn_var","pq_w","pq_b","conv_in_w","conv_in_b","norm_out_w","norm_out_b","conv_out_w","conv_out_b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        self.mid_r0.deinit(); self.mid_attn.deinit(); self.mid_r1.deinit();
        for (&self.up_resnets) |*blk| for (blk) |*r| r.deinit();
        for (0..3) |i| { _ = mlx.mlx_array_free(self.up_conv_w[i]); _ = mlx.mlx_array_free(self.up_conv_b[i]); }
    }

    /// packed_latents [1,128,64,64] (NCHW) -> image [1,3,1024,1024] (NCHW, [-1,1]).
    pub fn decode(self: *Vae, packed_in: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        // bn denorm: packed * sqrt(var+eps) + mean   (per-channel over 128)
        const vsh = [_]c_int{ 1, 128, 1, 1 };
        const bnm = try reshape(self.bn_mean, &vsh, s); defer _ = mlx.mlx_array_free(bnm);
        const bnv = try reshape(self.bn_var, &vsh, s); defer _ = mlx.mlx_array_free(bnv);
        const eps = mlx.mlx_array_new_float(1e-4); defer _ = mlx.mlx_array_free(eps);
        var vpe = mlx.mlx_array_new(); defer _ = mlx.mlx_array_free(vpe);
        try mlx.check(mlx.mlx_add(&vpe, bnv, eps, s));
        var std_ = mlx.mlx_array_new(); defer _ = mlx.mlx_array_free(std_);
        try mlx.check(mlx.mlx_sqrt(&std_, vpe, s));
        const pf = try astype(packed_in, .float32, s); defer _ = mlx.mlx_array_free(pf);
        const stdf = try astype(std_, .float32, s); defer _ = mlx.mlx_array_free(stdf);
        const bnmf = try astype(bnm, .float32, s); defer _ = mlx.mlx_array_free(bnmf);
        const scaled = try mulA(pf, stdf, s); defer _ = mlx.mlx_array_free(scaled);
        const denorm = try addA(scaled, bnmf, s); defer _ = mlx.mlx_array_free(denorm);
        // unpatchify [1,128,64,64] -> [1,32,128,128]
        const up = try unpatchify(denorm, s); defer _ = mlx.mlx_array_free(up);
        const up_bf = try astype(up, .bfloat16, s); defer _ = mlx.mlx_array_free(up_bf);
        // post_quant_conv (1x1) in NHWC
        const nhwc = try transpose(up_bf, &[_]c_int{ 0, 2, 3, 1 }, s); defer _ = mlx.mlx_array_free(nhwc);
        var h = try conv2d(nhwc, self.pq_w, self.pq_b, 0, s);
        // conv_in (3x3, 32->512)
        {
            const nh = try conv2d(h, self.conv_in_w, self.conv_in_b, 1, s);
            _ = mlx.mlx_array_free(h); h = nh;
        }
        // mid block
        { const nh = try self.mid_r0.forward(h, s); _ = mlx.mlx_array_free(h); h = nh; }
        { const nh = try self.mid_attn.forward(h, s); _ = mlx.mlx_array_free(h); h = nh; }
        { const nh = try self.mid_r1.forward(h, s); _ = mlx.mlx_array_free(h); h = nh; }
        // up blocks (3 resnets each; upsample on blocks 0,1,2)
        for (0..4) |bi| {
            for (0..3) |ri| {
                const nh = try self.up_resnets[bi][ri].forward(h, s);
                _ = mlx.mlx_array_free(h); h = nh;
            }
            if (bi < 3) {
                const us = try self.upsample(h, self.up_conv_w[bi], self.up_conv_b[bi], s);
                _ = mlx.mlx_array_free(h); h = us;
            }
        }
        // conv_norm_out + silu + conv_out
        { const nh = try groupNorm(h, self.norm_out_w, self.norm_out_b, 32, s); _ = mlx.mlx_array_free(h); h = nh; }
        { const nh = try silu(h, s); _ = mlx.mlx_array_free(h); h = nh; }
        { const nh = try conv2d(h, self.conv_out_w, self.conv_out_b, 1, s); _ = mlx.mlx_array_free(h); h = nh; }
        // NHWC -> NCHW (materialize: a lazy transpose reads back in source order)
        const out = try transpose(h, &[_]c_int{ 0, 3, 1, 2 }, s);
        _ = mlx.mlx_array_free(h);
        defer _ = mlx.mlx_array_free(out);
        var contig = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_contiguous(&contig, out, false, s));
        return contig;
    }

    fn upsample(self: *Vae, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
        _ = self;
        // x NHWC; nearest 2x on H,W via repeat, then conv3x3
        var r1 = mlx.mlx_array_new(); defer _ = mlx.mlx_array_free(r1);
        try mlx.check(mlx.mlx_repeat_axis(&r1, x, 2, 1, s));
        var r2 = mlx.mlx_array_new(); defer _ = mlx.mlx_array_free(r2);
        try mlx.check(mlx.mlx_repeat_axis(&r2, r1, 2, 2, s));
        return conv2d(r2, w, b, 1, s);
    }
};

/// [1,128,64,64] -> [1,32,128,128]  (reshape [1,32,2,2,64,64], transpose, reshape)
/// packed [1,128,gh,gw] → [1,32,gh·2,gw·2] (channel = c·4 + ph·2 + pw).
fn unpatchify(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const gh = sh[2];
    const gw = sh[3];
    const r1 = try reshape(x, &[_]c_int{ 1, 32, 2, 2, gh, gw }, s); defer _ = mlx.mlx_array_free(r1);
    const t1 = try transpose(r1, &[_]c_int{ 0, 1, 4, 2, 5, 3 }, s); defer _ = mlx.mlx_array_free(t1);
    return reshape(t1, &[_]c_int{ 1, 32, gh * 2, gw * 2 }, s);
}

fn loadResnet(w: *const Weights, a: std.mem.Allocator, pfx: []const u8) !Resnet {
    const g = struct {
        fn k(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8) !mlx.mlx_array {
            const kk = try fmtKey(aa, "{s}.{s}", .{ p, sub }); defer aa.free(kk);
            return ownWeight(ww, kk);
        }
        fn opt(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8) ?mlx.mlx_array {
            const kk = fmtKey(aa, "{s}.{s}", .{ p, sub }) catch return null; defer aa.free(kk);
            return ownOpt(ww, kk);
        }
    };
    return .{
        .n1w = try g.k(w, a, pfx, "norm1.weight"), .n1b = try g.k(w, a, pfx, "norm1.bias"),
        .c1w = try g.k(w, a, pfx, "conv1.weight"), .c1b = try g.k(w, a, pfx, "conv1.bias"),
        .n2w = try g.k(w, a, pfx, "norm2.weight"), .n2b = try g.k(w, a, pfx, "norm2.bias"),
        .c2w = try g.k(w, a, pfx, "conv2.weight"), .c2b = try g.k(w, a, pfx, "conv2.bias"),
        .sw = g.opt(w, a, pfx, "conv_shortcut.weight"), .sb = g.opt(w, a, pfx, "conv_shortcut.bias"),
    };
}

pub fn loadVae(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !Vae {
    const dir = try fmtKey(allocator, "{s}/vae", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    var v: Vae = undefined;
    v.allocator = allocator; v.s = s;
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
        const wk = try fmtKey(allocator, "decoder.up_blocks.{d}.upsamplers.0.conv.weight", .{bi}); defer allocator.free(wk);
        const bk = try fmtKey(allocator, "decoder.up_blocks.{d}.upsamplers.0.conv.bias", .{bi}); defer allocator.free(bk);
        v.up_conv_w[bi] = try ownWeight(&w, wk);
        v.up_conv_b[bi] = try ownWeight(&w, bk);
    }
    v.norm_out_w = try ownWeight(&w, "decoder.conv_norm_out.weight");
    v.norm_out_b = try ownWeight(&w, "decoder.conv_norm_out.bias");
    v.conv_out_w = try ownWeight(&w, "decoder.conv_out.weight");
    v.conv_out_b = try ownWeight(&w, "decoder.conv_out.bias");
    return v;
}

/// Asymmetric (0,1,0,1) zero-pad + 3x3 stride-2 valid conv on NHWC
/// (diffusers Downsample2D — the VAE encoder's spatial downsampler).
fn conv2dDown(x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, s: S) !mlx.mlx_array {
    const axes = [_]c_int{ 1, 2 };
    const low = [_]c_int{ 0, 0 };
    const high = [_]c_int{ 1, 1 };
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p);
    try mlx.check(mlx.mlx_pad(&p, x, &axes, 2, &low, 2, &high, 2, zero, "constant", s));
    var pc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pc);
    try mlx.check(mlx.mlx_contiguous(&pc, p, false, s));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, pc, w, 2, 2, 0, 0, 1, 1, 1, s));
    const r = try addA(o, bias, s);
    _ = mlx.mlx_array_free(o);
    return r;
}

/// Pixel-unshuffle NCHW [1,32,128,128] → packed [1,128,64,64]; exact inverse
/// of `unpatchify` (packed channel = c*4 + ph*2 + pw).
/// Pixel-unshuffle [1,32,H,W] → packed [1,128,H/2,W/2]; exact inverse of
/// `unpatchify`.
fn patchify(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const gh = @divExact(sh[2], 2);
    const gw = @divExact(sh[3], 2);
    const r1 = try reshape(x, &[_]c_int{ 1, 32, gh, 2, gw, 2 }, s);
    defer _ = mlx.mlx_array_free(r1);
    const t1 = try transpose(r1, &[_]c_int{ 0, 1, 3, 5, 2, 4 }, s);
    defer _ = mlx.mlx_array_free(t1);
    return reshape(t1, &[_]c_int{ 1, 128, gh, gw }, s);
}

/// FLUX.2 VAE encoder — structural mirror of the decoder (down_blocks with
/// stride-2 downsamplers instead of up_blocks). Feeds img2img: pixels →
/// bn-normalized packed latents, the exact space `Vae.decode` reads.
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
        for (&self.down_resnets) |*blk| for (blk) |*r| r.deinit();
        for (0..3) |i| {
            _ = mlx.mlx_array_free(self.down_conv_w[i]);
            _ = mlx.mlx_array_free(self.down_conv_b[i]);
        }
        self.mid_r0.deinit();
        self.mid_attn.deinit();
        self.mid_r1.deinit();
    }

    /// image [1,3,1024,1024] f32 [0,1] (NCHW) → packed latents [1,128,64,64]
    /// f32, bn-NORMALIZED (the distribution mean; deterministic — no sampling).
    pub fn encode(self: *VaeEncoder, img: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        // [0,1] → [-1,1], NCHW → NHWC bf16
        const two = mlx.mlx_array_new_float(2.0);
        defer _ = mlx.mlx_array_free(two);
        const one = mlx.mlx_array_new_float(1.0);
        defer _ = mlx.mlx_array_free(one);
        const x2 = try mulA(img, two, s);
        defer _ = mlx.mlx_array_free(x2);
        const xm = try subA(x2, one, s);
        defer _ = mlx.mlx_array_free(xm);
        const nhwc = try transpose(xm, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer _ = mlx.mlx_array_free(nhwc);
        const xbf = try astype(nhwc, .bfloat16, s);
        defer _ = mlx.mlx_array_free(xbf);

        var h = try conv2d(xbf, self.conv_in_w, self.conv_in_b, 1, s);
        for (0..4) |bi| {
            for (0..2) |ri| {
                const nh = try self.down_resnets[bi][ri].forward(h, s);
                _ = mlx.mlx_array_free(h);
                h = nh;
            }
            if (bi < 3) {
                const nh = try conv2dDown(h, self.down_conv_w[bi], self.down_conv_b[bi], s);
                _ = mlx.mlx_array_free(h);
                h = nh;
            }
        }
        {
            const nh = try self.mid_r0.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try self.mid_attn.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        {
            const nh = try self.mid_r1.forward(h, s);
            _ = mlx.mlx_array_free(h);
            h = nh;
        }
        const hn = try groupNorm(h, self.norm_out_w, self.norm_out_b, 32, s);
        _ = mlx.mlx_array_free(h);
        defer _ = mlx.mlx_array_free(hn);
        const ha = try silu(hn, s);
        defer _ = mlx.mlx_array_free(ha);
        const co = try conv2d(ha, self.conv_out_w, self.conv_out_b, 1, s); // [1,128,128,64]
        defer _ = mlx.mlx_array_free(co);
        const qc = try conv2d(co, self.quant_w, self.quant_b, 0, s);
        defer _ = mlx.mlx_array_free(qc);
        // mean = first 32 channels (NHWC last axis); logvar discarded.
        const qsh = mlx.getShape(qc);
        const start = [_]c_int{ 0, 0, 0, 0 };
        const stop = [_]c_int{ qsh[0], qsh[1], qsh[2], 32 };
        const strides = [_]c_int{ 1, 1, 1, 1 };
        var mean = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mean);
        try mlx.check(mlx.mlx_slice(&mean, qc, &start, 4, &stop, 4, &strides, 4, s));
        var meanc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(meanc);
        try mlx.check(mlx.mlx_contiguous(&meanc, mean, false, s));
        const nchw = try transpose(meanc, &[_]c_int{ 0, 3, 1, 2 }, s);
        defer _ = mlx.mlx_array_free(nchw);
        const nf = try astype(nchw, .float32, s);
        defer _ = mlx.mlx_array_free(nf);
        const packed_lat = try patchify(nf, s); // [1,128,64,64]
        defer _ = mlx.mlx_array_free(packed_lat);
        // bn normalize: (x - mean) / sqrt(var + eps)  (inverse of decode's denorm)
        const vsh = [_]c_int{ 1, 128, 1, 1 };
        const bnm = try reshape(self.bn_mean, &vsh, s);
        defer _ = mlx.mlx_array_free(bnm);
        const bnv = try reshape(self.bn_var, &vsh, s);
        defer _ = mlx.mlx_array_free(bnv);
        const eps = mlx.mlx_array_new_float(1e-4);
        defer _ = mlx.mlx_array_free(eps);
        var vpe = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(vpe);
        try mlx.check(mlx.mlx_add(&vpe, bnv, eps, s));
        var std_ = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(std_);
        try mlx.check(mlx.mlx_sqrt(&std_, vpe, s));
        const stdf = try astype(std_, .float32, s);
        defer _ = mlx.mlx_array_free(stdf);
        const bnmf = try astype(bnm, .float32, s);
        defer _ = mlx.mlx_array_free(bnmf);
        const centered = try subA(packed_lat, bnmf, s);
        defer _ = mlx.mlx_array_free(centered);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_divide(&out, centered, stdf, s));
        return out;
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

/// Conditioning rebalance: scale the prompt embeddings by a global `gain` and
/// optional per-tap `weights` (len 3 — the encoder's captured layers 9/18/27,
/// concatenated on the feature axis). Returns a NEW array in the input dtype.
pub fn applyCondRebalance(enc: mlx.mlx_array, gain: f32, weights: ?[]const f32, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(enc); // [1,seq,3H]
    const dt = mlx.mlx_array_dtype(enc);
    var cur = blk: {
        var c = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&c, enc));
        break :blk c;
    };
    if (weights) |w| {
        if (w.len != 3) {
            _ = mlx.mlx_array_free(cur);
            return error.InvalidCondWeights;
        }
        const H = @divExact(sh[2], 3);
        const r = try reshape(cur, &[_]c_int{ sh[0], sh[1], 3, H }, s);
        _ = mlx.mlx_array_free(cur);
        const wsh = [_]c_int{ 1, 1, 3, 1 };
        const wa = mlx.mlx_array_new_data(w.ptr, &wsh, 4, .float32);
        defer _ = mlx.mlx_array_free(wa);
        const scaled = mulA(r, wa, s) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        _ = mlx.mlx_array_free(r);
        cur = try reshape(scaled, &[_]c_int{ sh[0], sh[1], sh[2] }, s);
        _ = mlx.mlx_array_free(scaled);
    }
    if (gain != 1.0) {
        const ga = mlx.mlx_array_new_float(gain);
        defer _ = mlx.mlx_array_free(ga);
        const scaled = mulA(cur, ga, s) catch |e| {
            _ = mlx.mlx_array_free(cur);
            return e;
        };
        _ = mlx.mlx_array_free(cur);
        cur = scaled;
    }
    if (mlx.mlx_array_dtype(cur) != dt) {
        const back = try astype(cur, dt, s);
        _ = mlx.mlx_array_free(cur);
        cur = back;
    }
    return cur;
}

// Stage 3 oracle: VAE decode packed_pre_vae -> decoded.
//   FLUX_TEST_MODEL, FLUX_PACKED (packed_pre_vae f32 [1,128,64,64]), FLUX_DECODED
test "flux VAE reproduces decoded image" {
    const model_dir = std.mem.span(std.c.getenv("FLUX_TEST_MODEL") orelse return error.SkipZigTest);
    const pk_p = std.mem.span(std.c.getenv("FLUX_PACKED") orelse return error.SkipZigTest);
    const dec_p = std.mem.span(std.c.getenv("FLUX_DECODED") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const pk = try readF32(io, a, pk_p); defer a.free(pk);
    const ref = try readF32(io, a, dec_p); defer a.free(ref);
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var vae = try loadVae(io, a, s, model_dir); defer vae.deinit();
    const pk_sh = [_]c_int{ 1, 128, 64, 64 };
    const pk_f = mlx.mlx_array_new_data(pk.ptr, &pk_sh, 4, .float32); defer _ = mlx.mlx_array_free(pk_f);
    const dec = try vae.decode(pk_f); defer _ = mlx.mlx_array_free(dec);
    const df = try astype(dec, .float32, s); defer _ = mlx.mlx_array_free(df);
    _ = mlx.mlx_array_eval(df);
    const n: usize = @intCast(mlx.mlx_array_size(df));
    const data = mlx.mlx_array_data_float32(df) orelse return error.NoData;
    try testing.expectEqual(ref.len, n);
    var dot: f64 = 0; var na: f64 = 0; var nb: f64 = 0; var se: f64 = 0;
    for (0..n) |i| { const x: f64 = data[i]; const y: f64 = ref[i]; dot += x*y; na += x*x; nb += y*y; se += (x-y)*(x-y); }
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
    const lat = try enc.encode(img);
    defer _ = mlx.mlx_array_free(lat);
    const lsh = mlx.getShape(lat);
    try testing.expectEqual(@as(c_int, 128), lsh[1]);
    try testing.expectEqual(@as(c_int, 64), lsh[2]);
    const dec = try vae.decode(lat); // [-1,1]
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
    const lat = try enc.encode(img);
    defer _ = mlx.mlx_array_free(lat);
    const lsh = mlx.getShape(lat);
    try testing.expectEqual(@as(c_int, 128), lsh[1]);
    try testing.expectEqual(@as(c_int, @intCast(H / 16)), lsh[2]);
    try testing.expectEqual(@as(c_int, @intCast(W / 16)), lsh[3]);
    const dec = try vae.decode(lat); // [-1,1]
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
    const ref_lat = try venc.encode(img);
    defer _ = mlx.mlx_array_free(ref_lat);

    const base = try generateWithOpts(&te, &dit, &vae, &ids, &mask, 42, 2, 1024, 1024, .{}, null);
    defer _ = mlx.mlx_array_free(base);
    const edited = try generateWithOpts(&te, &dit, &vae, &ids, &mask, 42, 2, 1024, 1024, .{ .ref_latents = ref_lat }, null);
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
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // enc [1,2,6]: hidden H=2 per tap, 3 taps concatenated on the feature axis.
    const vals = [_]f32{ 1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60 };
    const sh = [_]c_int{ 1, 2, 6 };
    const enc = mlx.mlx_array_new_data(&vals, &sh, 3, .float32);
    defer _ = mlx.mlx_array_free(enc);
    const w = [_]f32{ 2, 3, 4 };
    const out = try applyCondRebalance(enc, 0.5, &w, s);
    defer _ = mlx.mlx_array_free(out);
    _ = mlx.mlx_array_eval(out);
    const d = mlx.mlx_array_data_float32(out) orelse return error.NoData;
    // Row 0: [1,2]*2*.5, [3,4]*3*.5, [5,6]*4*.5 = [1,2, 4.5,6, 10,12]
    const expect0 = [_]f32{ 1, 2, 4.5, 6, 10, 12 };
    for (expect0, 0..) |e, i| try testing.expectApproxEqAbs(e, d[i], 1e-5);
    // Gain-only path (weights null) multiplies everything.
    const g = try applyCondRebalance(enc, 2.0, null, s);
    defer _ = mlx.mlx_array_free(g);
    _ = mlx.mlx_array_eval(g);
    const gd = mlx.mlx_array_data_float32(g) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 2), gd[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 120), gd[11], 1e-5);
    // Wrong count is a hard error (3 taps for FLUX).
    const bad = [_]f32{ 1, 2 };
    try testing.expectError(error.InvalidCondWeights, applyCondRebalance(enc, 1.0, &bad, s));
}

// ════════════════════════════════════════════════════════════════════════
// Full text→image pipeline.
// ════════════════════════════════════════════════════════════════════════

/// Empirical mu + sigma/timestep schedule (FlowMatchEuler, image_seq_len, steps).
fn computeSchedule(allocator: std.mem.Allocator, image_seq_len: u32, steps: u32) !struct { ts: []f32, sig: []f32 } {
    const ts = try allocator.alloc(f32, steps);
    const sig = try allocator.alloc(f32, steps + 1);
    const isl: f64 = @floatFromInt(image_seq_len);
    const a1 = 8.73809524e-05; const b1 = 1.89833333;
    const a2 = 0.00016927; const b2 = 0.45666666;
    var mu: f64 = undefined;
    if (isl > 4300) { mu = a2 * isl + b2; } else {
        const m_200 = a2 * isl + b2;
        const m_10 = a1 * isl + b1;
        const a = (m_200 - m_10) / 190.0;
        const b = m_200 - 200.0 * a;
        mu = a * @as(f64, @floatFromInt(steps)) + b;
    }
    const emu = std.math.exp(mu);
    for (0..steps) |i| {
        const sl = 1.0 - (@as(f64, @floatFromInt(i)) * (1.0 - 1.0 / @as(f64, @floatFromInt(steps))) / @as(f64, @floatFromInt(steps - 1)));
        const shifted = emu / (emu + (1.0 / sl - 1.0));
        sig[i] = @floatCast(shifted);
        ts[i] = @floatCast(shifted * 1000.0);
    }
    sig[steps] = 0.0;
    return .{ .ts = ts, .sig = sig };
}

fn buildLatentIds(allocator: std.mem.Allocator, lh: u32, lw: u32) ![]i32 {
    return buildLatentIdsT(allocator, lh, lw, 0);
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
    /// concatenated on the token sequence with t=10 position ids. Never
    /// noised; generation starts from pure noise and attends to them.
    ref_latents: ?mlx.mlx_array = null,
    /// Conditioning rebalance: global gain + per-tap weights (len 3).
    cond_gain: f32 = 1.0,
    cond_weights: ?[]const f32 = null,
};

pub fn generate(te: *TextEncoder, dit: *Dit, vae: *Vae, ids: []const i32, mask: []const i32, seed: u64, steps: u32, height: u32, width: u32, progress: ?sse.Progress) !mlx.mlx_array {
    return generateWithOpts(te, dit, vae, ids, mask, seed, steps, height, width, .{}, progress);
}

/// Stage 1 alone: encode the prompt (+ optional conditioning rebalance) and
/// MATERIALIZE the result. The eval matters: mlx is lazy, and a low-memory
/// host (the iPhone) frees the 1.7 GB text encoder right after this returns —
/// an unevaluated graph would pin the encoder weights via refcounts and the
/// free would reclaim nothing.
pub fn encodePrompt(te: *TextEncoder, ids: []const i32, mask: []const i32, opts: GenOpts) !mlx.mlx_array {
    var enc = try te.encode(ids, mask);
    if (opts.cond_gain != 1.0 or opts.cond_weights != null) {
        const re = try applyCondRebalance(enc, opts.cond_gain, opts.cond_weights, te.s);
        _ = mlx.mlx_array_free(enc);
        enc = re;
    }
    _ = mlx.mlx_array_eval(enc);
    return enc;
}

pub fn generateWithOpts(te: *TextEncoder, dit: *Dit, vae: *Vae, ids: []const i32, mask: []const i32, seed: u64, steps: u32, height: u32, width: u32, opts: GenOpts, progress: ?sse.Progress) !mlx.mlx_array {
    const enc = try encodePrompt(te, ids, mask, opts);
    return generateFromCondWithOpts(dit, vae, enc, ids.len, seed, steps, height, width, opts, progress);
}

/// Stages 2+ (latents init → denoise loop → VAE decode) from a pre-computed
/// prompt encoding. Takes ownership of `enc_owned`. `prompt_len` = token count
/// of the encoded prompt (drives the text position ids).
pub fn generateFromCondWithOpts(dit: *Dit, vae: *Vae, enc_owned: mlx.mlx_array, prompt_len: usize, seed: u64, steps: u32, height: u32, width: u32, opts: GenOpts, progress: ?sse.Progress) !mlx.mlx_array {
    const s = dit.s;
    const a = dit.allocator;
    const lh = height / 16;
    const lw = width / 16;
    const nlat = lh * lw;

    const enc = enc_owned;
    defer _ = mlx.mlx_array_free(enc);

    // 2. latents init: random.normal([1,128,lh,lw], key(seed)) → pack [1,nlat,128]
    var key = mlx.mlx_array_new(); defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    const nsh = [_]c_int{ 1, 128, @intCast(lh), @intCast(lw) };
    var noise = mlx.mlx_array_new(); defer _ = mlx.mlx_array_free(noise);
    try mlx.check(mlx.mlx_random_normal(&noise, &nsh, 4, .float32, 0.0, 1.0, key, s));
    const noise_bf = try astype(noise, .bfloat16, s); defer _ = mlx.mlx_array_free(noise_bf);
    // pack: [1,128,nlat] transpose [0,2,1]
    const r = try reshape(noise_bf, &[_]c_int{ 1, 128, @intCast(nlat) }, s); defer _ = mlx.mlx_array_free(r);
    var latents = try transpose(r, &[_]c_int{ 0, 2, 1 }, s); // [1,nlat,128]
    { var c = mlx.mlx_array_new(); try mlx.check(mlx.mlx_contiguous(&c, latents, false, s)); _ = mlx.mlx_array_free(latents); latents = c; }

    const img_ids = try buildLatentIds(a, lh, lw); defer a.free(img_ids);
    const txt_ids = try buildTextIds(a, @intCast(prompt_len)); defer a.free(txt_ids);

    // Reference-image editing: pack the CLEAN reference latents into tokens
    // with t=10 position ids (official FLUX.2 convention). They ride along in
    // every forward; only the generated span is Euler-stepped.
    var ref_tokens: mlx.mlx_array = .{ .ctx = null };
    defer if (ref_tokens.ctx != null) {
        _ = mlx.mlx_array_free(ref_tokens);
    };
    var all_ids: ?[]i32 = null;
    defer if (all_ids) |ai| a.free(ai);
    if (opts.ref_latents) |rl| {
        const rsh = mlx.getShape(rl); // [1,128,rh,rw]
        const rlh: u32 = @intCast(rsh[2]);
        const rlw: u32 = @intCast(rsh[3]);
        const rr = try reshape(rl, &[_]c_int{ 1, 128, @intCast(rlh * rlw) }, s);
        defer _ = mlx.mlx_array_free(rr);
        const rt = try transpose(rr, &[_]c_int{ 0, 2, 1 }, s);
        defer _ = mlx.mlx_array_free(rt);
        var rtc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rtc);
        try mlx.check(mlx.mlx_contiguous(&rtc, rt, false, s));
        ref_tokens = try astype(rtc, .bfloat16, s);
        const rids = try buildLatentIdsT(a, rlh, rlw, 10);
        defer a.free(rids);
        const ai = try a.alloc(i32, img_ids.len + rids.len);
        @memcpy(ai[0..img_ids.len], img_ids);
        @memcpy(ai[img_ids.len..], rids);
        all_ids = ai;
    }

    // 3. denoise loop
    const sched = try computeSchedule(a, nlat, steps); defer { a.free(sched.ts); a.free(sched.sig); }
    const start_step: u32 = @min(opts.start_step, steps - 1);

    // img2img: x = (1-σ)·z0 + σ·noise at the start sigma (flow-match scale_noise).
    if (opts.init_latents) |z0| {
        const zr = try reshape(z0, &[_]c_int{ 1, 128, @intCast(nlat) }, s);
        defer _ = mlx.mlx_array_free(zr);
        const zt = try transpose(zr, &[_]c_int{ 0, 2, 1 }, s);
        defer _ = mlx.mlx_array_free(zt);
        const z_bf = try astype(zt, .bfloat16, s);
        defer _ = mlx.mlx_array_free(z_bf);
        const sigma = sched.sig[start_step];
        const sa = mlx.mlx_array_new_float(sigma);
        defer _ = mlx.mlx_array_free(sa);
        const oma = mlx.mlx_array_new_float(1.0 - sigma);
        defer _ = mlx.mlx_array_free(oma);
        const zs = try mulA(z_bf, oma, s);
        defer _ = mlx.mlx_array_free(zs);
        const ns = try mulA(latents, sa, s);
        defer _ = mlx.mlx_array_free(ns);
        const mixed = try addA(zs, ns, s);
        _ = mlx.mlx_array_free(latents);
        latents = try astype(mixed, .bfloat16, s);
        _ = mlx.mlx_array_free(mixed);
    }

    const run_steps = steps - start_step;
    for (start_step..steps) |t| {
        const nz = blk: {
            if (ref_tokens.ctx != null) {
                const input = try concat(&[_]mlx.mlx_array{ latents, ref_tokens }, 1, s);
                defer _ = mlx.mlx_array_free(input);
                const full = try dit.forward(input, enc, sched.ts[t], all_ids.?, txt_ids);
                defer _ = mlx.mlx_array_free(full);
                break :blk try slice3(full, 1, 0, @intCast(nlat), s);
            }
            break :blk try dit.forward(latents, enc, sched.ts[t], img_ids, txt_ids);
        };
        defer _ = mlx.mlx_array_free(nz);
        const dt = sched.sig[t + 1] - sched.sig[t];
        const dta = mlx.mlx_array_new_float(dt); defer _ = mlx.mlx_array_free(dta);
        const step = try mulA(nz, dta, s); defer _ = mlx.mlx_array_free(step);
        const nl = try addA(latents, step, s);
        _ = mlx.mlx_array_free(latents); latents = nl;
        _ = mlx.mlx_array_eval(latents);
        if (progress) |p| p.emit("Generating", @intCast(t + 1 - start_step), run_steps);
    }
    if (progress) |p| p.emit("Decoding image", steps, steps);

    // 4. unpack → [1,128,lh,lw], decode
    const lr = try reshape(latents, &[_]c_int{ 1, @intCast(lh), @intCast(lw), 128 }, s); defer _ = mlx.mlx_array_free(lr);
    _ = mlx.mlx_array_free(latents);
    const packed_lat = try transpose(lr, &[_]c_int{ 0, 3, 1, 2 }, s); defer _ = mlx.mlx_array_free(packed_lat);
    const decoded = try vae.decode(packed_lat); defer _ = mlx.mlx_array_free(decoded);
    // denormalize: clip(x/2+0.5, 0, 1)
    const half = mlx.mlx_array_new_float(0.5); defer _ = mlx.mlx_array_free(half);
    const df = try astype(decoded, .float32, s); defer _ = mlx.mlx_array_free(df);
    const scaled = try mulA(df, half, s); defer _ = mlx.mlx_array_free(scaled);
    const shifted = try addA(scaled, half, s); defer _ = mlx.mlx_array_free(shifted);
    const lo = mlx.mlx_array_new_float(0.0); defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_float(1.0); defer _ = mlx.mlx_array_free(hi);
    var clo = mlx.mlx_array_new(); defer _ = mlx.mlx_array_free(clo);
    try mlx.check(mlx.mlx_maximum(&clo, shifted, lo, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_minimum(&out, clo, hi, s));
    return out;
}

/// Write [1,3,H,W] f32 [0,1] as a binary PPM (P6) — viewable, no deps.
pub fn writePpm(io: std.Io, allocator: std.mem.Allocator, img: mlx.mlx_array, path: []const u8, s: S) !void {
    const cf = blk: { var c = mlx.mlx_array_new(); try mlx.check(mlx.mlx_contiguous(&c, img, false, s)); break :blk c; };
    defer _ = mlx.mlx_array_free(cf);
    _ = mlx.mlx_array_eval(cf);
    const sh = mlx.getShape(cf); // [1,3,H,W]
    const H: usize = @intCast(sh[2]); const W: usize = @intCast(sh[3]);
    const d = mlx.mlx_array_data_float32(cf) orelse return error.NoData;
    const hdr = try std.fmt.allocPrint(allocator, "P6\n{d} {d}\n255\n", .{ W, H });
    defer allocator.free(hdr);
    const body = try allocator.alloc(u8, W * H * 3);
    defer allocator.free(body);
    const plane = W * H;
    for (0..H) |y| for (0..W) |x| {
        const o = (y * W + x) * 3;
        for (0..3) |c| {
            const v = d[c * plane + y * W + x];
            body[o + c] = @intFromFloat(std.math.clamp(v * 255.0, 0, 255));
        }
    };
    const f = try std.Io.Dir.createFileAbsolute(io, path, .{});
    defer f.close(io);
    var wb: [4096]u8 = undefined;
    var fw = f.writer(io, &wb);
    try fw.interface.writeAll(hdr);
    try fw.interface.writeAll(body);
    try fw.interface.flush();
}

// End-to-end: full pipeline vs the reference image (and writes a PPM).
//   FLUX_TEST_MODEL, FLUX_IDS, FLUX_MASK, FLUX_DECODED(decoded.raw), FLUX_PPM(out, optional)
test "flux end-to-end pipeline matches reference image" {
    const model_dir = std.mem.span(std.c.getenv("FLUX_TEST_MODEL") orelse return error.SkipZigTest);
    const ids_p = std.mem.span(std.c.getenv("FLUX_IDS") orelse return error.SkipZigTest);
    const mask_p = std.mem.span(std.c.getenv("FLUX_MASK") orelse return error.SkipZigTest);
    const dec_p = std.mem.span(std.c.getenv("FLUX_DECODED") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ids = try readI32(io, a, ids_p); defer a.free(ids);
    const mask = try readI32(io, a, mask_p); defer a.free(mask);
    const ref = try readF32(io, a, dec_p); defer a.free(ref); // [1,3,1024,1024] in [-1,1]

    const s = mlx.mlx_default_gpu_stream_new(); defer _ = mlx.mlx_stream_free(s);
    var te = try loadTextEncoder(io, a, s, model_dir); defer te.deinit();
    var dit = try loadDit(io, a, s, model_dir); defer dit.deinit();
    var vae = try loadVae(io, a, s, model_dir); defer vae.deinit();

    const img = try generate(&te, &dit, &vae, ids, mask, 42, 4, 1024, 1024, null); // [0,1]
    defer _ = mlx.mlx_array_free(img);
    _ = mlx.mlx_array_eval(img);
    const n: usize = @intCast(mlx.mlx_array_size(img));
    const data = mlx.mlx_array_data_float32(img) orelse return error.NoData;
    // reference is in [-1,1]; convert to [0,1] for comparison
    var dot: f64 = 0; var na: f64 = 0; var nb: f64 = 0;
    for (0..n) |i| {
        const x: f64 = data[i];
        const y: f64 = std.math.clamp(@as(f64, ref[i]) / 2.0 + 0.5, 0, 1);
        dot += x * y; na += x * x; nb += y * y;
    }
    const corr = dot / (std.math.sqrt(na) * std.math.sqrt(nb));
    std.debug.print("[flux-e2e] n={d} corr={d:.5}\n", .{ n, corr });
    if (std.c.getenv("FLUX_PPM")) |pp| try writePpm(io, a, img, std.mem.span(pp), s);
    try testing.expect(corr > 0.95);
}

// ── Tests ──

test "hermetic conv2d sanity" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // input [1,4,4,2] all ones, weight [3,3,3,2] all ones (out=3,kh=3,kw=3,in=2), pad=1
    var inb: [1 * 4 * 4 * 2]f32 = undefined; @memset(&inb, 1.0);
    const ish = [_]c_int{ 1, 4, 4, 2 };
    const inp = mlx.mlx_array_new_data(&inb, &ish, 4, .float32); defer _ = mlx.mlx_array_free(inp);
    var wb: [3 * 3 * 3 * 2]f32 = undefined; @memset(&wb, 1.0);
    const wsh = [_]c_int{ 3, 3, 3, 2 };
    const w = mlx.mlx_array_new_data(&wb, &wsh, 4, .float32); defer _ = mlx.mlx_array_free(w);
    var o = mlx.mlx_array_new(); defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_conv2d(&o, inp, w, 1, 1, 1, 1, 1, 1, 1, s));
    _ = mlx.mlx_array_eval(o);
    const sh = mlx.getShape(o);
    std.debug.print("[conv-sanity] out shape [{d},{d},{d},{d}]\n", .{ sh[0], sh[1], sh[2], sh[3] });
    const d = mlx.mlx_array_data_float32(o).?;
    // center pixel (1,1) channel0 = full 3x3x2 window of ones = 18; corner = 2x2x2=8
    std.debug.print("[conv-sanity] center={d} corner={d}\n", .{ d[(1 * 4 + 1) * 3 + 0], d[0] });
    try testing.expect(@abs(d[(1 * 4 + 1) * 3 + 0] - 18.0) < 0.01);
    // bf16 variant
    const in_bf = try astype(inp, .bfloat16, s); defer _ = mlx.mlx_array_free(in_bf);
    const w_bf = try astype(w, .bfloat16, s); defer _ = mlx.mlx_array_free(w_bf);
    var o2 = mlx.mlx_array_new(); defer _ = mlx.mlx_array_free(o2);
    try mlx.check(mlx.mlx_conv2d(&o2, in_bf, w_bf, 1, 1, 1, 1, 1, 1, 1, s));
    const o2f = try astype(o2, .float32, s); defer _ = mlx.mlx_array_free(o2f);
    _ = mlx.mlx_array_eval(o2f);
    const d2 = mlx.mlx_array_data_float32(o2f).?;
    std.debug.print("[conv-sanity-bf16] center={d} corner={d}\n", .{ d2[(1 * 4 + 1) * 3 + 0], d2[0] });
}

const testing = std.testing;

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

    const pe = try te.encode(ids, mask);
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

        const got = try ql.forward(x, s);
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
