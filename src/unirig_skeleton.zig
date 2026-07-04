//! UniRig stage-1 SKELETON engine — image/mesh point cloud → autoregressive
//! skeleton-token sequence → joints/parents/tails. Phase 3 of the 3D pipeline,
//! sibling of `src/hunyuan3d.zig` (whose engine style, MixedLinear primitive,
//! safetensors loader, and env-gated cos-oracle test harness this mirrors).
//!
//! Two sub-models over one `skeleton.safetensors` (contract
//! `tests/unirig_weights_contract.md`):
//!   - `enc.*`  a clean-room michelangelo perceiver: Fourier-embed the point
//!     cloud + normals, cross-attend 1024 FPS query latents into the full cloud,
//!     16 self-attention blocks, ln_post → [1,1024,512] latents.
//!   - `ar.*`   a stock OPT-350m decoder (24 pre-norm layers, ReLU FFN, learned
//!     absolute positions with +2 offset, top-level final_layer_norm). The 1024
//!     latents are lifted by `output_proj` (512→1024) and prepended as a soft
//!     prefix ([mesh…, bos, cls]); generation is grammar-masked greedy/sampled
//!     decode (grammar + detokenize live in `unirig_tokenizer.zig`).
//!
//! The michelangelo encoder is re-derived clean-room (its reference source is
//! GPLv3); only the converted MIT weights cross. Its fused c_qkv/c_kv per-head
//! interleave is BAKED OUT at convert time, so the head reshapes here are
//! STANDARD (never re-interleave — same discipline as hunyuan3d.zig DitAttn).

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const hy3d = @import("hunyuan3d.zig");
const utok = @import("unirig_tokenizer.zig");
const fps = @import("fps.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;
const MixedLinear = hy3d.MixedLinear;

/// OPT + torch nn.LayerNorm default eps (NOT hunyuan3d's 1e-6 conv/DINO eps).
const LN_EPS: f32 = 1e-5;

// ── file-local mlx wrappers (mirror the non-pub hunyuan3d.zig primitives) ──
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
/// [B,L,H*hd] → [B,H,L,hd]
fn splitHeads(x: mlx.mlx_array, heads: c_int, hd: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const x4 = try reshape(x, &[_]c_int{ sh[0], sh[1], heads, hd }, s);
    defer _ = mlx.mlx_array_free(x4);
    return transpose(x4, &[_]c_int{ 0, 2, 1, 3 }, s);
}
/// [B,H,L,hd] → [B,L,H*hd]
fn mergeHeads(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const t = try transpose(x, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(t);
    return reshape(t, &[_]c_int{ sh[0], sh[2], sh[1] * sh[3] }, s);
}
/// SDPA. `causal` selects the mask mode ("causal" for the OPT decoder self-attn,
/// "" bidirectional for the michelangelo encoder). q/k/v [B,H,L,hd].
fn sdpa(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, causal: bool, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    const null_a = mlx.mlx_array{ .ctx = null };
    const mode: [*:0]const u8 = if (causal) "causal" else "";
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, mode, null_a, null_a, s));
    return o;
}
/// Exact (erf) GELU — michelangelo MLPs use torch nn.GELU() (approximate="none").
fn geluErf(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const inv_sqrt2 = mlx.mlx_array_new_float(0.7071067811865476);
    defer _ = mlx.mlx_array_free(inv_sqrt2);
    const xs = try mulA(x, inv_sqrt2, s);
    defer _ = mlx.mlx_array_free(xs);
    var e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e);
    try mlx.check(mlx.mlx_erf(&e, xs, s));
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    const opt = try addA(e, one, s);
    defer _ = mlx.mlx_array_free(opt);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const hx = try mulA(x, half, s);
    defer _ = mlx.mlx_array_free(hx);
    return mulA(hx, opt, s);
}
/// ReLU — OPT FFN activation.
fn reluA(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_maximum(&o, x, zero, s));
    return o;
}
/// q from `xq`, k/v from `xkv`; standard head reshape; single-op attention.
fn attention(
    q_l: *const MixedLinear,
    k_l: *const MixedLinear,
    v_l: *const MixedLinear,
    o_l: *const MixedLinear,
    xq: mlx.mlx_array,
    xkv: mlx.mlx_array,
    heads: c_int,
    hd: c_int,
    causal: bool,
    s: S,
) !mlx.mlx_array {
    const q0 = try q_l.forward(xq, s);
    defer _ = mlx.mlx_array_free(q0);
    const k0 = try k_l.forward(xkv, s);
    defer _ = mlx.mlx_array_free(k0);
    const v0 = try v_l.forward(xkv, s);
    defer _ = mlx.mlx_array_free(v0);
    const q = try splitHeads(q0, heads, hd, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try splitHeads(k0, heads, hd, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try splitHeads(v0, heads, hd, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(@as(u32, @intCast(hd)))));
    const attn = try sdpa(q, k, v, scale, causal, s);
    defer _ = mlx.mlx_array_free(attn);
    const merged = try mergeHeads(attn, s);
    defer _ = mlx.mlx_array_free(merged);
    return o_l.forward(merged, s);
}

// ── config ──────────────────────────────────────────────────────────────────

pub const SkelConfig = struct {
    // AR OPT decoder
    ar_layers: u32 = 24,
    ar_hidden: u32 = 1024,
    ar_heads: u32 = 16,
    ar_ffn: u32 = 4096,
    pos_offset: u32 = 2,
    vocab: u32 = 267,
    // michelangelo encoder
    enc_width: u32 = 512,
    enc_heads: u32 = 8,
    enc_layers: u32 = 16,
    num_freqs: u32 = 8,
    token_num: u32 = 1024,
    // tokenizer default cls (articulationxl)
    default_cls: u16 = utok.Tok.cls_articulationxl,

    pub fn arHeadDim(self: SkelConfig) u32 {
        return self.ar_hidden / self.ar_heads; // 64
    }
    pub fn encHeadDim(self: SkelConfig) u32 {
        return self.enc_width / self.enc_heads; // 64
    }
};

/// Validate + (loosely) parse config.json. The architecture is fixed, so this
/// asserts model_type and returns the baked defaults (mirrors how hunyuan3d.zig
/// keys its arch off a config it also mostly hardcodes).
pub fn parseConfigText(text: []const u8) !SkelConfig {
    if (std.mem.indexOf(u8, text, "unirig_skeleton") == null) {
        log.err("[unirig] config.json model_type is not unirig_skeleton\n", .{});
        return error.BadUnirigConfig;
    }
    return .{};
}

fn readConfigFile(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !SkelConfig {
    const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(path);
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const text = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    defer allocator.free(text);
    return parseConfigText(text);
}

// ── michelangelo perceiver encoder ────────────────────────────────────────────

const CrossBlock = struct {
    ln1_w: mlx.mlx_array, // query norm
    ln1_b: mlx.mlx_array,
    ln2_w: mlx.mlx_array, // kv norm
    ln2_b: mlx.mlx_array,
    q: MixedLinear,
    k: MixedLinear,
    v: MixedLinear,
    o: MixedLinear,
    ln3_w: mlx.mlx_array,
    ln3_b: mlx.mlx_array,
    fc1: MixedLinear,
    fc2: MixedLinear,

    fn deinit(self: *CrossBlock) void {
        for ([_]mlx.mlx_array{ self.ln1_w, self.ln1_b, self.ln2_w, self.ln2_b, self.ln3_w, self.ln3_b }) |a| _ = mlx.mlx_array_free(a);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        self.fc1.deinit();
        self.fc2.deinit();
    }
};

const SelfBlock = struct {
    ln1_w: mlx.mlx_array,
    ln1_b: mlx.mlx_array,
    q: MixedLinear,
    k: MixedLinear,
    v: MixedLinear,
    o: MixedLinear,
    ln2_w: mlx.mlx_array,
    ln2_b: mlx.mlx_array,
    fc1: MixedLinear,
    fc2: MixedLinear,

    fn deinit(self: *SelfBlock) void {
        for ([_]mlx.mlx_array{ self.ln1_w, self.ln1_b, self.ln2_w, self.ln2_b }) |a| _ = mlx.mlx_array_free(a);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        self.fc1.deinit();
        self.fc2.deinit();
    }
};

pub const Encoder = struct {
    cfg: SkelConfig,
    allocator: std.mem.Allocator,
    s: S,
    input_proj: MixedLinear,
    cross: CrossBlock,
    blocks: []SelfBlock,
    ln_post_w: mlx.mlx_array,
    ln_post_b: mlx.mlx_array,

    fn load(w: *const Weights, alloc: std.mem.Allocator, cfg: SkelConfig, s: S) !Encoder {
        const width = cfg.enc_width;
        const inter = width * 4;
        var self: Encoder = undefined;
        self.cfg = cfg;
        self.allocator = alloc;
        self.s = s;
        self.input_proj = try MixedLinear.load(w, alloc, "enc.input_proj", 3 * (2 * cfg.num_freqs + 1) + 3, s);

        // cross block
        self.cross = .{
            .ln1_w = try hy3d.normVec(w, "enc.cross_attn.ln1.weight", s),
            .ln1_b = try hy3d.normVec(w, "enc.cross_attn.ln1.bias", s),
            .ln2_w = try hy3d.normVec(w, "enc.cross_attn.ln2.weight", s),
            .ln2_b = try hy3d.normVec(w, "enc.cross_attn.ln2.bias", s),
            .q = try MixedLinear.load(w, alloc, "enc.cross_attn.attn.q", width, s),
            .k = try MixedLinear.load(w, alloc, "enc.cross_attn.attn.k", width, s),
            .v = try MixedLinear.load(w, alloc, "enc.cross_attn.attn.v", width, s),
            .o = try MixedLinear.load(w, alloc, "enc.cross_attn.attn.out", width, s),
            .ln3_w = try hy3d.normVec(w, "enc.cross_attn.ln3.weight", s),
            .ln3_b = try hy3d.normVec(w, "enc.cross_attn.ln3.bias", s),
            .fc1 = try MixedLinear.load(w, alloc, "enc.cross_attn.mlp.fc1", width, s),
            .fc2 = try MixedLinear.load(w, alloc, "enc.cross_attn.mlp.fc2", inter, s),
        };

        self.blocks = try alloc.alloc(SelfBlock, cfg.enc_layers);
        for (self.blocks, 0..) |*blk, i| {
            const p_ln1w = try hy3d.fmtKey(alloc, "enc.blocks.{d}.ln1.weight", .{i});
            defer alloc.free(p_ln1w);
            const p_ln1b = try hy3d.fmtKey(alloc, "enc.blocks.{d}.ln1.bias", .{i});
            defer alloc.free(p_ln1b);
            const p_ln2w = try hy3d.fmtKey(alloc, "enc.blocks.{d}.ln2.weight", .{i});
            defer alloc.free(p_ln2w);
            const p_ln2b = try hy3d.fmtKey(alloc, "enc.blocks.{d}.ln2.bias", .{i});
            defer alloc.free(p_ln2b);
            const p_q = try hy3d.fmtKey(alloc, "enc.blocks.{d}.attn.q", .{i});
            defer alloc.free(p_q);
            const p_k = try hy3d.fmtKey(alloc, "enc.blocks.{d}.attn.k", .{i});
            defer alloc.free(p_k);
            const p_v = try hy3d.fmtKey(alloc, "enc.blocks.{d}.attn.v", .{i});
            defer alloc.free(p_v);
            const p_o = try hy3d.fmtKey(alloc, "enc.blocks.{d}.attn.out", .{i});
            defer alloc.free(p_o);
            const p_f1 = try hy3d.fmtKey(alloc, "enc.blocks.{d}.mlp.fc1", .{i});
            defer alloc.free(p_f1);
            const p_f2 = try hy3d.fmtKey(alloc, "enc.blocks.{d}.mlp.fc2", .{i});
            defer alloc.free(p_f2);
            blk.* = .{
                .ln1_w = try hy3d.normVec(w, p_ln1w, s),
                .ln1_b = try hy3d.normVec(w, p_ln1b, s),
                .q = try MixedLinear.load(w, alloc, p_q, width, s),
                .k = try MixedLinear.load(w, alloc, p_k, width, s),
                .v = try MixedLinear.load(w, alloc, p_v, width, s),
                .o = try MixedLinear.load(w, alloc, p_o, width, s),
                .ln2_w = try hy3d.normVec(w, p_ln2w, s),
                .ln2_b = try hy3d.normVec(w, p_ln2b, s),
                .fc1 = try MixedLinear.load(w, alloc, p_f1, width, s),
                .fc2 = try MixedLinear.load(w, alloc, p_f2, inter, s),
            };
        }
        self.ln_post_w = try hy3d.normVec(w, "enc.ln_post.weight", s);
        self.ln_post_b = try hy3d.normVec(w, "enc.ln_post.bias", s);
        return self;
    }

    fn deinit(self: *Encoder) void {
        self.input_proj.deinit();
        self.cross.deinit();
        for (self.blocks) |*b| b.deinit();
        self.allocator.free(self.blocks);
        _ = mlx.mlx_array_free(self.ln_post_w);
        _ = mlx.mlx_array_free(self.ln_post_b);
    }

    /// Fourier-embed a [1,P,3] point set + its [1,P,3] normals → input_proj → [1,P,512].
    fn embedPoints(self: *const Encoder, pc: mlx.mlx_array, normals: mlx.mlx_array, s: S) !mlx.mlx_array {
        const fe = try hy3d.fourierEmbed(pc, self.cfg.num_freqs, s); // [1,P,51]
        defer _ = mlx.mlx_array_free(fe);
        const cat = try concat(&[_]mlx.mlx_array{ fe, normals }, 2, s); // [1,P,54]
        defer _ = mlx.mlx_array_free(cat);
        return self.input_proj.forward(cat, s); // [1,P,512] f16
    }

    /// pc/normals [1,N,3] f32; `q_idx` are indices into N for the 1024 FPS query
    /// points. Returns latents [1,1024,512] f16. KV = the FULL cloud
    /// (use_full_input=true); Q = the FPS points. Bidirectional attention.
    pub fn encodeLatents(self: *const Encoder, pc: mlx.mlx_array, normals: mlx.mlx_array, q_idx: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const heads: c_int = @intCast(self.cfg.enc_heads);
        const hd: c_int = @intCast(self.cfg.encHeadDim());

        const data = try self.embedPoints(pc, normals, s); // KV [1,N,512]
        defer _ = mlx.mlx_array_free(data);

        // gather the 1024 query points + their normals
        var q_pc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_pc);
        try mlx.check(mlx.mlx_take_axis(&q_pc, pc, q_idx, 1, s)); // [1,1024,3]
        var q_no = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_no);
        try mlx.check(mlx.mlx_take_axis(&q_no, normals, q_idx, 1, s));
        const sampled = try self.embedPoints(q_pc, q_no, s); // Q [1,1024,512]
        defer _ = mlx.mlx_array_free(sampled);

        // cross block (pre-LN): x = x + attn(ln1(x), ln2(data)); x = x + mlp(ln3(x))
        const cn_q = try layerNorm(sampled, self.cross.ln1_w, self.cross.ln1_b, s);
        defer _ = mlx.mlx_array_free(cn_q);
        const cn_kv = try layerNorm(data, self.cross.ln2_w, self.cross.ln2_b, s);
        defer _ = mlx.mlx_array_free(cn_kv);
        const ca = try attention(&self.cross.q, &self.cross.k, &self.cross.v, &self.cross.o, cn_q, cn_kv, heads, hd, false, s);
        defer _ = mlx.mlx_array_free(ca);
        var x = try addA(sampled, ca, s);
        {
            const n3 = try layerNorm(x, self.cross.ln3_w, self.cross.ln3_b, s);
            defer _ = mlx.mlx_array_free(n3);
            const f1 = try self.cross.fc1.forward(n3, s);
            defer _ = mlx.mlx_array_free(f1);
            const g = try geluErf(f1, s);
            defer _ = mlx.mlx_array_free(g);
            const f2 = try self.cross.fc2.forward(g, s);
            defer _ = mlx.mlx_array_free(f2);
            const nx = try addA(x, f2, s);
            _ = mlx.mlx_array_free(x);
            x = nx;
        }

        // 16 self-attention blocks (pre-LN)
        for (self.blocks) |*blk| {
            const n1 = try layerNorm(x, blk.ln1_w, blk.ln1_b, s);
            defer _ = mlx.mlx_array_free(n1);
            const a = try attention(&blk.q, &blk.k, &blk.v, &blk.o, n1, n1, heads, hd, false, s);
            defer _ = mlx.mlx_array_free(a);
            const h1 = try addA(x, a, s);
            _ = mlx.mlx_array_free(x);
            const n2 = try layerNorm(h1, blk.ln2_w, blk.ln2_b, s);
            defer _ = mlx.mlx_array_free(n2);
            const f1 = try blk.fc1.forward(n2, s);
            defer _ = mlx.mlx_array_free(f1);
            const g = try geluErf(f1, s);
            defer _ = mlx.mlx_array_free(g);
            const f2 = try blk.fc2.forward(g, s);
            defer _ = mlx.mlx_array_free(f2);
            x = try addA(h1, f2, s);
            _ = mlx.mlx_array_free(h1);
        }
        defer _ = mlx.mlx_array_free(x);
        return layerNorm(x, self.ln_post_w, self.ln_post_b, s); // [1,1024,512]
    }
};

// ── OPT-350m decoder ──────────────────────────────────────────────────────────

const OptLayer = struct {
    attn_norm_w: mlx.mlx_array,
    attn_norm_b: mlx.mlx_array,
    q: MixedLinear,
    k: MixedLinear,
    v: MixedLinear,
    o: MixedLinear,
    mlp_norm_w: mlx.mlx_array,
    mlp_norm_b: mlx.mlx_array,
    fc1: MixedLinear,
    fc2: MixedLinear,

    fn deinit(self: *OptLayer) void {
        for ([_]mlx.mlx_array{ self.attn_norm_w, self.attn_norm_b, self.mlp_norm_w, self.mlp_norm_b }) |a| _ = mlx.mlx_array_free(a);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        self.fc1.deinit();
        self.fc2.deinit();
    }
};

/// Per-layer dense K/V cache for incremental greedy decode. Each entry is
/// [1,heads,T,hd] and grows by concat along the T axis; materialized each step so
/// the lazy graph doesn't compound across the decode. The no-cache `forward` stays
/// the oracle reference; `forwardCached` must reproduce it (pinned by the
/// cache-vs-no-cache equivalence test).
const KvCache = struct {
    k: []mlx.mlx_array,
    v: []mlx.mlx_array,

    fn init(alloc: std.mem.Allocator, n: usize) !KvCache {
        const k = try alloc.alloc(mlx.mlx_array, n);
        errdefer alloc.free(k);
        const v = try alloc.alloc(mlx.mlx_array, n);
        for (k) |*a| a.* = .{ .ctx = null };
        for (v) |*a| a.* = .{ .ctx = null };
        return .{ .k = k, .v = v };
    }
    fn deinit(self: *KvCache, alloc: std.mem.Allocator) void {
        for (self.k) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        for (self.v) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        alloc.free(self.k);
        alloc.free(self.v);
    }
};

pub const Decoder = struct {
    cfg: SkelConfig,
    allocator: std.mem.Allocator,
    s: S,
    embed_tokens: mlx.mlx_array, // [267,1024] f16
    embed_positions: mlx.mlx_array, // [3078,1024] f16
    layers: []OptLayer,
    final_norm_w: mlx.mlx_array,
    final_norm_b: mlx.mlx_array,
    lm_head: MixedLinear,

    fn loadEmbed(w: *const Weights, key: []const u8, s: S) !mlx.mlx_array {
        const raw = try hy3d.ownWeight(w, key);
        defer _ = mlx.mlx_array_free(raw);
        return astype(raw, .float16, s);
    }

    fn load(w: *const Weights, alloc: std.mem.Allocator, cfg: SkelConfig, s: S) !Decoder {
        const H = cfg.ar_hidden;
        var self: Decoder = undefined;
        self.cfg = cfg;
        self.allocator = alloc;
        self.s = s;
        self.embed_tokens = try loadEmbed(w, "ar.embed_tokens", s);
        self.embed_positions = try loadEmbed(w, "ar.embed_positions", s);
        self.layers = try alloc.alloc(OptLayer, cfg.ar_layers);
        for (self.layers, 0..) |*layer, i| {
            const p_anw = try hy3d.fmtKey(alloc, "ar.layers.{d}.attn_norm.weight", .{i});
            defer alloc.free(p_anw);
            const p_anb = try hy3d.fmtKey(alloc, "ar.layers.{d}.attn_norm.bias", .{i});
            defer alloc.free(p_anb);
            const p_q = try hy3d.fmtKey(alloc, "ar.layers.{d}.attn.q", .{i});
            defer alloc.free(p_q);
            const p_k = try hy3d.fmtKey(alloc, "ar.layers.{d}.attn.k", .{i});
            defer alloc.free(p_k);
            const p_v = try hy3d.fmtKey(alloc, "ar.layers.{d}.attn.v", .{i});
            defer alloc.free(p_v);
            const p_o = try hy3d.fmtKey(alloc, "ar.layers.{d}.attn.out", .{i});
            defer alloc.free(p_o);
            const p_mnw = try hy3d.fmtKey(alloc, "ar.layers.{d}.mlp_norm.weight", .{i});
            defer alloc.free(p_mnw);
            const p_mnb = try hy3d.fmtKey(alloc, "ar.layers.{d}.mlp_norm.bias", .{i});
            defer alloc.free(p_mnb);
            const p_f1 = try hy3d.fmtKey(alloc, "ar.layers.{d}.mlp.fc1", .{i});
            defer alloc.free(p_f1);
            const p_f2 = try hy3d.fmtKey(alloc, "ar.layers.{d}.mlp.fc2", .{i});
            defer alloc.free(p_f2);
            layer.* = .{
                .attn_norm_w = try hy3d.normVec(w, p_anw, s),
                .attn_norm_b = try hy3d.normVec(w, p_anb, s),
                .q = try MixedLinear.load(w, alloc, p_q, H, s),
                .k = try MixedLinear.load(w, alloc, p_k, H, s),
                .v = try MixedLinear.load(w, alloc, p_v, H, s),
                .o = try MixedLinear.load(w, alloc, p_o, H, s),
                .mlp_norm_w = try hy3d.normVec(w, p_mnw, s),
                .mlp_norm_b = try hy3d.normVec(w, p_mnb, s),
                .fc1 = try MixedLinear.load(w, alloc, p_f1, H, s),
                .fc2 = try MixedLinear.load(w, alloc, p_f2, cfg.ar_ffn, s),
            };
        }
        self.final_norm_w = try hy3d.normVec(w, "ar.final_norm.weight", s);
        self.final_norm_b = try hy3d.normVec(w, "ar.final_norm.bias", s);
        self.lm_head = try MixedLinear.load(w, alloc, "ar.lm_head", H, s);
        return self;
    }

    fn deinit(self: *Decoder) void {
        _ = mlx.mlx_array_free(self.embed_tokens);
        _ = mlx.mlx_array_free(self.embed_positions);
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        _ = mlx.mlx_array_free(self.final_norm_w);
        _ = mlx.mlx_array_free(self.final_norm_b);
        self.lm_head.deinit();
    }

    /// Gather token embeddings for `ids` → [1,n,1024] f16.
    fn embedTokens(self: *const Decoder, ids: []const u16, s: S) !mlx.mlx_array {
        const idx = try self.allocator.alloc(i32, ids.len);
        defer self.allocator.free(idx);
        for (ids, 0..) |v, i| idx[i] = @intCast(v);
        const ish = [_]c_int{@intCast(ids.len)};
        const id_arr = mlx.mlx_array_new_data(idx.ptr, &ish, 1, .int32);
        defer _ = mlx.mlx_array_free(id_arr);
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_axis(&taken, self.embed_tokens, id_arr, 0, s));
        return reshape(taken, &[_]c_int{ 1, @intCast(ids.len), @intCast(self.cfg.ar_hidden) }, s);
    }

    /// Full-sequence causal forward over `inputs_embeds` [1,L,1024] (mesh prefix +
    /// token embeds). Adds OPT learned absolute positions (offset +2), runs 24
    /// pre-norm layers + top-level final_norm, projects to vocab. Returns logits
    /// [1,L,267] f16. (No KV cache — decode re-forwards the growing sequence;
    /// KV-cache reuse is a follow-up perf item.)
    pub fn forward(self: *const Decoder, inputs_embeds: mlx.mlx_array, s: S) !mlx.mlx_array {
        const heads: c_int = @intCast(self.cfg.ar_heads);
        const hd: c_int = @intCast(self.cfg.arHeadDim());
        const H: c_int = @intCast(self.cfg.ar_hidden);
        const L: usize = @intCast(mlx.getShape(inputs_embeds)[1]);

        // learned absolute positions: rows [offset .. offset+L)
        const pos_idx = try self.allocator.alloc(i32, L);
        defer self.allocator.free(pos_idx);
        for (0..L) |i| pos_idx[i] = @intCast(self.cfg.pos_offset + i);
        const psh = [_]c_int{@intCast(L)};
        const pos_arr = mlx.mlx_array_new_data(pos_idx.ptr, &psh, 1, .int32);
        defer _ = mlx.mlx_array_free(pos_arr);
        var pos_rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos_rows);
        try mlx.check(mlx.mlx_take_axis(&pos_rows, self.embed_positions, pos_arr, 0, s));
        const pos = try reshape(pos_rows, &[_]c_int{ 1, @intCast(L), H }, s);
        defer _ = mlx.mlx_array_free(pos);
        var h = try addA(inputs_embeds, pos, s);

        for (self.layers) |*layer| {
            const n1 = try layerNorm(h, layer.attn_norm_w, layer.attn_norm_b, s);
            defer _ = mlx.mlx_array_free(n1);
            const a = try attention(&layer.q, &layer.k, &layer.v, &layer.o, n1, n1, heads, hd, true, s);
            defer _ = mlx.mlx_array_free(a);
            const h1 = try addA(h, a, s);
            _ = mlx.mlx_array_free(h);
            const n2 = try layerNorm(h1, layer.mlp_norm_w, layer.mlp_norm_b, s);
            defer _ = mlx.mlx_array_free(n2);
            const f1 = try layer.fc1.forward(n2, s);
            defer _ = mlx.mlx_array_free(f1);
            const r = try reluA(f1, s);
            defer _ = mlx.mlx_array_free(r);
            const f2 = try layer.fc2.forward(r, s);
            defer _ = mlx.mlx_array_free(f2);
            h = try addA(h1, f2, s);
            _ = mlx.mlx_array_free(h1);
        }
        const hn = try layerNorm(h, self.final_norm_w, self.final_norm_b, s);
        _ = mlx.mlx_array_free(h);
        defer _ = mlx.mlx_array_free(hn);
        return self.lm_head.forward(hn, s); // [1,L,267]
    }

    /// Incremental forward with a growing per-layer K/V cache. `start_pos` is the
    /// absolute position of the FIRST token in `inputs_embeds` (0 for the prefill,
    /// then the running cache length). Appends this call's K/V to the cache and
    /// attends over the whole cache. Prefill (L>1) is causal; a single decode
    /// token attends to all cached keys (it is at the sequence end). Returns
    /// logits [1,L,267]. Numerically equivalent to `forward` on the concatenated
    /// sequence (pinned by the equivalence test).
    pub fn forwardCached(self: *const Decoder, inputs_embeds: mlx.mlx_array, cache: *KvCache, start_pos: usize, s: S) !mlx.mlx_array {
        const heads: c_int = @intCast(self.cfg.ar_heads);
        const hd: c_int = @intCast(self.cfg.arHeadDim());
        const H: c_int = @intCast(self.cfg.ar_hidden);
        const L: usize = @intCast(mlx.getShape(inputs_embeds)[1]);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(self.cfg.arHeadDim())));
        const causal = L > 1;

        const pos_idx = try self.allocator.alloc(i32, L);
        defer self.allocator.free(pos_idx);
        for (0..L) |i| pos_idx[i] = @intCast(self.cfg.pos_offset + start_pos + i);
        const psh = [_]c_int{@intCast(L)};
        const pos_arr = mlx.mlx_array_new_data(pos_idx.ptr, &psh, 1, .int32);
        defer _ = mlx.mlx_array_free(pos_arr);
        var pos_rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos_rows);
        try mlx.check(mlx.mlx_take_axis(&pos_rows, self.embed_positions, pos_arr, 0, s));
        const pos = try reshape(pos_rows, &[_]c_int{ 1, @intCast(L), H }, s);
        defer _ = mlx.mlx_array_free(pos);
        var h = try addA(inputs_embeds, pos, s);

        for (self.layers, 0..) |*layer, li| {
            const n1 = try layerNorm(h, layer.attn_norm_w, layer.attn_norm_b, s);
            defer _ = mlx.mlx_array_free(n1);
            const q0 = try layer.q.forward(n1, s);
            defer _ = mlx.mlx_array_free(q0);
            const k0 = try layer.k.forward(n1, s);
            defer _ = mlx.mlx_array_free(k0);
            const v0 = try layer.v.forward(n1, s);
            defer _ = mlx.mlx_array_free(v0);
            const q = try splitHeads(q0, heads, hd, s);
            defer _ = mlx.mlx_array_free(q);
            const knew = try splitHeads(k0, heads, hd, s); // [1,heads,L,hd]
            const vnew = try splitHeads(v0, heads, hd, s);
            // append to (or seed) the cache
            if (cache.k[li].ctx == null) {
                cache.k[li] = knew;
                cache.v[li] = vnew;
            } else {
                const kf = try concat(&[_]mlx.mlx_array{ cache.k[li], knew }, 2, s);
                const vf = try concat(&[_]mlx.mlx_array{ cache.v[li], vnew }, 2, s);
                _ = mlx.mlx_array_free(cache.k[li]);
                _ = mlx.mlx_array_free(cache.v[li]);
                _ = mlx.mlx_array_free(knew);
                _ = mlx.mlx_array_free(vnew);
                cache.k[li] = kf;
                cache.v[li] = vf;
            }
            _ = mlx.mlx_array_eval(cache.k[li]); // materialize; keep the graph flat
            _ = mlx.mlx_array_eval(cache.v[li]);
            const attn = try sdpa(q, cache.k[li], cache.v[li], scale, causal, s);
            defer _ = mlx.mlx_array_free(attn);
            const merged = try mergeHeads(attn, s);
            defer _ = mlx.mlx_array_free(merged);
            const a = try layer.o.forward(merged, s);
            defer _ = mlx.mlx_array_free(a);
            const h1 = try addA(h, a, s);
            _ = mlx.mlx_array_free(h);
            const n2 = try layerNorm(h1, layer.mlp_norm_w, layer.mlp_norm_b, s);
            defer _ = mlx.mlx_array_free(n2);
            const f1 = try layer.fc1.forward(n2, s);
            defer _ = mlx.mlx_array_free(f1);
            const r = try reluA(f1, s);
            defer _ = mlx.mlx_array_free(r);
            const f2 = try layer.fc2.forward(r, s);
            defer _ = mlx.mlx_array_free(f2);
            h = try addA(h1, f2, s);
            _ = mlx.mlx_array_free(h1);
        }
        const hn = try layerNorm(h, self.final_norm_w, self.final_norm_b, s);
        _ = mlx.mlx_array_free(h);
        defer _ = mlx.mlx_array_free(hn);
        return self.lm_head.forward(hn, s);
    }
};

// ── AR-normalize + surface sample (pure; hermetically testable) ───────────────

/// AugmentAffine normalize-into-[-1,1] transform: isotropic (single max-extent)
/// scale about the bbox center. `apply`: n = (p − center)·scale;
/// `inverse`: p = n / scale + center. Normals are unchanged (translate + uniform
/// positive scale preserves direction).
pub const NormXform = struct {
    center: [3]f32,
    scale: f32,

    pub fn compute(pts: []const f32) NormXform {
        var lo = [3]f32{ std.math.floatMax(f32), std.math.floatMax(f32), std.math.floatMax(f32) };
        var hi = [3]f32{ -std.math.floatMax(f32), -std.math.floatMax(f32), -std.math.floatMax(f32) };
        var i: usize = 0;
        while (i < pts.len) : (i += 3) {
            for (0..3) |c| {
                lo[c] = @min(lo[c], pts[i + c]);
                hi[c] = @max(hi[c], pts[i + c]);
            }
        }
        var max_ext: f32 = 0;
        var center: [3]f32 = undefined;
        for (0..3) |c| {
            center[c] = (lo[c] + hi[c]) * 0.5;
            max_ext = @max(max_ext, hi[c] - lo[c]);
        }
        // scale into [-1,1] (extent hi−lo = 2); guard degenerate/point clouds.
        const scale: f32 = if (max_ext > 1e-9) 2.0 / max_ext else 1.0;
        return .{ .center = center, .scale = scale };
    }
    pub fn applyInPlace(self: NormXform, pts: []f32) void {
        var i: usize = 0;
        while (i < pts.len) : (i += 3) {
            for (0..3) |c| pts[i + c] = (pts[i + c] - self.center[c]) * self.scale;
        }
    }
    pub fn inverseJoint(self: NormXform, j: [3]f32) [3]f32 {
        return .{
            j[0] / self.scale + self.center[0],
            j[1] / self.scale + self.center[1],
            j[2] / self.scale + self.center[2],
        };
    }
};

/// Area-weighted barycentric surface sampling of `num` points + interpolated unit
/// normals from a triangle mesh (positions V×3, normals V×3, faces F×3). Fixed
/// seed → deterministic (an accepted divergence from the reference numpy-PCG64
/// SamplerMix; dossier §8 LOW). Falls back to subsampling the raw vertices when
/// no faces are given. Caller owns the returned slices.
const Sampled = struct { pts: []f32, nrm: []f32 };
fn sampleSurface(alloc: std.mem.Allocator, positions: []const f32, normals: []const f32, faces: []const u32, num: usize, seed: u64) !Sampled {
    const pts = try alloc.alloc(f32, num * 3);
    errdefer alloc.free(pts);
    const nrm = try alloc.alloc(f32, num * 3);
    errdefer alloc.free(nrm);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    if (faces.len < 3) {
        // no connectivity → sample vertices (with replacement if num > V)
        const nv = positions.len / 3;
        for (0..num) |k| {
            const vi = rng.intRangeLessThan(usize, 0, nv);
            for (0..3) |c| {
                pts[k * 3 + c] = positions[vi * 3 + c];
                nrm[k * 3 + c] = normals[vi * 3 + c];
            }
        }
        return .{ .pts = pts, .nrm = nrm };
    }

    const nf = faces.len / 3;
    const cum = try alloc.alloc(f32, nf);
    defer alloc.free(cum);
    var total: f32 = 0;
    for (0..nf) |t| {
        const va = faces[t * 3 + 0];
        const vb = faces[t * 3 + 1];
        const vc = faces[t * 3 + 2];
        var e1: [3]f32 = undefined;
        var e2: [3]f32 = undefined;
        for (0..3) |c| {
            e1[c] = positions[vb * 3 + c] - positions[va * 3 + c];
            e2[c] = positions[vc * 3 + c] - positions[va * 3 + c];
        }
        const cx = e1[1] * e2[2] - e1[2] * e2[1];
        const cy = e1[2] * e2[0] - e1[0] * e2[2];
        const cz = e1[0] * e2[1] - e1[1] * e2[0];
        total += 0.5 * @sqrt(cx * cx + cy * cy + cz * cz);
        cum[t] = total;
    }
    if (total <= 1e-12) return error.DegenerateMesh;

    for (0..num) |k| {
        // pick a triangle proportional to area (binary search the cumulative area)
        const r = rng.float(f32) * total;
        var lo_i: usize = 0;
        var hi_i: usize = nf - 1;
        while (lo_i < hi_i) {
            const mid = (lo_i + hi_i) / 2;
            if (cum[mid] < r) lo_i = mid + 1 else hi_i = mid;
        }
        const t = lo_i;
        // uniform barycentric (reflect into the lower triangle)
        var u = rng.float(f32);
        var v = rng.float(f32);
        if (u + v > 1.0) {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        const w = 1.0 - u - v;
        const va = faces[t * 3 + 0];
        const vb = faces[t * 3 + 1];
        const vc = faces[t * 3 + 2];
        var n: [3]f32 = undefined;
        for (0..3) |c| {
            pts[k * 3 + c] = w * positions[va * 3 + c] + u * positions[vb * 3 + c] + v * positions[vc * 3 + c];
            n[c] = w * normals[va * 3 + c] + u * normals[vb * 3 + c] + v * normals[vc * 3 + c];
        }
        const nl = @sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        const inv: f32 = if (nl > 1e-9) 1.0 / nl else 1.0;
        for (0..3) |c| nrm[k * 3 + c] = n[c] * inv;
    }
    return .{ .pts = pts, .nrm = nrm };
}

/// Deterministic presample of `count` distinct indices from `n` (seeded partial
/// Fisher-Yates). Not numpy-PCG64 (accepted parity gap; dossier §8 LOW).
fn presampleIndices(alloc: std.mem.Allocator, n: usize, count: usize, seed: u64) ![]u32 {
    const m = @min(count, n);
    const perm = try alloc.alloc(u32, n);
    defer alloc.free(perm);
    for (0..n) |i| perm[i] = @intCast(i);
    var prng = std.Random.DefaultPrng.init(seed ^ 0x9E3779B97F4A7C15);
    const rng = prng.random();
    for (0..m) |i| {
        const j = rng.intRangeLessThan(usize, i, n);
        const tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
    const out = try alloc.alloc(u32, m);
    @memcpy(out, perm[0..m]);
    return out;
}

/// Grammar-masked argmax over the vocab (greedy next token).
fn pickNextToken(ids: []const u16, logits: []const f32, mask: *[utok.Tok.vocab_size]bool) u16 {
    utok.nextPossibleTokens(ids, mask);
    var best: f32 = -std.math.inf(f32);
    var best_i: u16 = utok.Tok.eos;
    for (0..utok.Tok.vocab_size) |i| {
        if (mask[i] and logits[i] > best) {
            best = logits[i];
            best_i = @intCast(i);
        }
    }
    return best_i;
}

// ── engine ────────────────────────────────────────────────────────────────────

pub const SkeletonOpts = struct {
    num_samples: u32 = 65536, // surface points sampled from the mesh (encoder KV cloud)
    cls: u16 = utok.Tok.cls_articulationxl,
    max_new: usize = 2048,
    seed: u64 = 12345,
    use_kv_cache: bool = true,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    s: S,
    cfg: SkelConfig,
    encoder: Encoder,
    decoder: Decoder,
    output_proj: MixedLinear, // 512 → 1024

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.s = mlx.mlx_default_gpu_stream_new();
        self.cfg = try readConfigFile(io, allocator, model_dir);
        var w = try hy3d.loadFileWeights(allocator, model_dir, "skeleton.safetensors");
        defer w.deinit();
        self.encoder = try Encoder.load(&w, allocator, self.cfg, self.s);
        errdefer self.encoder.deinit();
        self.decoder = try Decoder.load(&w, allocator, self.cfg, self.s);
        errdefer self.decoder.deinit();
        self.output_proj = try MixedLinear.load(&w, allocator, "output_proj", self.cfg.enc_width, self.s);
        log.info("[unirig] skeleton engine ready\n", .{});
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.encoder.deinit();
        self.decoder.deinit();
        self.output_proj.deinit();
        self.allocator.destroy(self);
    }

    /// Assemble the decoder input prefix embeds: output_proj(latents) followed by
    /// the [bos, cls] token embeddings → [1,1026,1024] f16. Positional embeddings
    /// are added later, inside Decoder.forward (matches the reference).
    fn buildPrefix(self: *Engine, latents: mlx.mlx_array, cls: u16) !mlx.mlx_array {
        const s = self.s;
        const mesh_tokens = try self.output_proj.forward(latents, s); // [1,1024,1024]
        defer _ = mlx.mlx_array_free(mesh_tokens);
        const start = try self.decoder.embedTokens(&[_]u16{ utok.Tok.bos, cls }, s); // [1,2,1024]
        defer _ = mlx.mlx_array_free(start);
        return concat(&[_]mlx.mlx_array{ mesh_tokens, start }, 1, s);
    }

    /// Grammar-masked GREEDY decode. `q_idx` picks the encoder query points.
    /// Returns the full token sequence [bos, cls, …, eos] (caller owns).
    pub fn generateTokensGreedy(
        self: *Engine,
        alloc: std.mem.Allocator,
        pc: mlx.mlx_array,
        normals: mlx.mlx_array,
        q_idx: mlx.mlx_array,
        cls: u16,
        max_new: usize,
    ) ![]u16 {
        const s = self.s;
        const latents = try self.encoder.encodeLatents(pc, normals, q_idx);
        defer _ = mlx.mlx_array_free(latents);
        const mesh_tokens = try self.output_proj.forward(latents, s); // [1,1024,1024]
        defer _ = mlx.mlx_array_free(mesh_tokens);

        var ids: std.ArrayList(u16) = .empty;
        errdefer ids.deinit(alloc);
        try ids.append(alloc, utok.Tok.bos);
        try ids.append(alloc, cls);

        var mask: [utok.Tok.vocab_size]bool = undefined;
        var step: usize = 0;
        while (step < max_new) : (step += 1) {
            // inputs_embeds = [mesh_tokens, embed(ids)]
            const tok_emb = try self.decoder.embedTokens(ids.items, s);
            defer _ = mlx.mlx_array_free(tok_emb);
            const seq = try concat(&[_]mlx.mlx_array{ mesh_tokens, tok_emb }, 1, s);
            defer _ = mlx.mlx_array_free(seq);
            const logits = try self.decoder.forward(seq, s); // [1,L,267]
            defer _ = mlx.mlx_array_free(logits);
            const L = mlx.getShape(logits)[1];
            const last = try sliceAxis(logits, 1, L - 1, L, s); // [1,1,267]
            defer _ = mlx.mlx_array_free(last);
            const lastf = try astype(last, .float32, s);
            defer _ = mlx.mlx_array_free(lastf);
            _ = mlx.mlx_array_eval(lastf);
            const data = mlx.mlx_array_data_float32(lastf) orelse return error.NoData;

            const best_i = pickNextToken(ids.items, data[0..utok.Tok.vocab_size], &mask);
            try ids.append(alloc, best_i);
            if (best_i == utok.Tok.eos) break;
        }
        return ids.toOwnedSlice(alloc);
    }

    /// Same greedy trajectory as `generateTokensGreedy` but with a per-layer K/V
    /// cache: prefill the [mesh, bos, cls] prefix once, then feed one token per
    /// step. O(n) instead of the no-cache O(n²) re-forward. Equivalence with the
    /// no-cache path is pinned by the cache-vs-no-cache test.
    pub fn generateTokensGreedyCached(
        self: *Engine,
        alloc: std.mem.Allocator,
        pc: mlx.mlx_array,
        normals: mlx.mlx_array,
        q_idx: mlx.mlx_array,
        cls: u16,
        max_new: usize,
    ) ![]u16 {
        const s = self.s;
        const latents = try self.encoder.encodeLatents(pc, normals, q_idx);
        defer _ = mlx.mlx_array_free(latents);
        const mesh_tokens = try self.output_proj.forward(latents, s); // [1,1024,1024]
        defer _ = mlx.mlx_array_free(mesh_tokens);

        var cache = try KvCache.init(alloc, self.cfg.ar_layers);
        defer cache.deinit(alloc);
        var ids: std.ArrayList(u16) = .empty;
        errdefer ids.deinit(alloc);
        try ids.append(alloc, utok.Tok.bos);
        try ids.append(alloc, cls);

        var mask: [utok.Tok.vocab_size]bool = undefined;
        // prefill: [mesh_tokens, embed(bos,cls)]
        const start_emb = try self.decoder.embedTokens(ids.items, s);
        defer _ = mlx.mlx_array_free(start_emb);
        const prefix = try concat(&[_]mlx.mlx_array{ mesh_tokens, start_emb }, 1, s);
        defer _ = mlx.mlx_array_free(prefix);
        var cache_len: usize = @intCast(mlx.getShape(prefix)[1]); // 1026 after prefill

        var logits = try self.decoder.forwardCached(prefix, &cache, 0, s);
        var step: usize = 0;
        while (step < max_new) : (step += 1) {
            const Lc = mlx.getShape(logits)[1];
            const last = try sliceAxis(logits, 1, Lc - 1, Lc, s);
            const lastf = try astype(last, .float32, s);
            _ = mlx.mlx_array_free(last);
            _ = mlx.mlx_array_eval(lastf);
            const data = mlx.mlx_array_data_float32(lastf) orelse {
                _ = mlx.mlx_array_free(lastf);
                _ = mlx.mlx_array_free(logits);
                return error.NoData;
            };
            const best_i = pickNextToken(ids.items, data[0..utok.Tok.vocab_size], &mask);
            _ = mlx.mlx_array_free(lastf);
            _ = mlx.mlx_array_free(logits);
            try ids.append(alloc, best_i);
            if (best_i == utok.Tok.eos) break;
            const step_emb = try self.decoder.embedTokens(&[_]u16{best_i}, s); // [1,1,1024]
            logits = try self.decoder.forwardCached(step_emb, &cache, cache_len, s);
            _ = mlx.mlx_array_free(step_emb);
            cache_len += 1;
        } else {
            _ = mlx.mlx_array_free(logits);
        }
        return ids.toOwnedSlice(alloc);
    }

    /// Full live rig path — the ONE function the gen.zig rig hook calls. Takes a
    /// mesh in the CALLER's coordinate frame (positions V×3, normals V×3, faces
    /// F×3 triangle indices) and returns the predicted skeleton (joints/parents/
    /// tails) in that SAME frame. Pipeline: area-weighted surface sample →
    /// AR normalize into the unit cube → seed-fixed presample + FPS query points →
    /// michelangelo encode → grammar-masked greedy AR decode → detokenize →
    /// inverse-normalize back to the caller's coordinates. Caller owns the result.
    pub fn generateSkeleton(
        self: *Engine,
        alloc: std.mem.Allocator,
        positions: []const f32,
        normals: []const f32,
        indices: []const u32,
        opts: SkeletonOpts,
    ) !utok.Skeleton {
        // 1. sample the surface into a point cloud + normals
        const sampled = try sampleSurface(alloc, positions, normals, indices, opts.num_samples, opts.seed);
        defer alloc.free(sampled.pts);
        defer alloc.free(sampled.nrm);
        // 2. normalize into the unit cube (records the inverse transform)
        const xform = NormXform.compute(sampled.pts);
        xform.applyInPlace(sampled.pts); // normals unchanged (uniform scale + translate)
        const n = sampled.pts.len / 3;

        // 3. presample + FPS query indices (into the full cloud)
        const pre = try presampleIndices(alloc, n, self.cfg.token_num * 4, opts.seed);
        defer alloc.free(pre);
        const pre_pts = try alloc.alloc(f32, pre.len * 3);
        defer alloc.free(pre_pts);
        for (pre, 0..) |gi, i| {
            for (0..3) |c| pre_pts[i * 3 + c] = sampled.pts[@as(usize, gi) * 3 + c];
        }
        const fps_local = try fps.farthestPointSample(alloc, pre_pts, self.cfg.token_num);
        defer alloc.free(fps_local);
        const q_abs = try alloc.alloc(i32, fps_local.len);
        defer alloc.free(q_abs);
        for (fps_local, 0..) |li, i| q_abs[i] = @intCast(pre[@as(usize, li)]);

        // 4. build mlx inputs
        const psh = [_]c_int{ 1, @intCast(n), 3 };
        const pc = mlx.mlx_array_new_data(sampled.pts.ptr, &psh, 3, .float32);
        defer _ = mlx.mlx_array_free(pc);
        const nrm = mlx.mlx_array_new_data(sampled.nrm.ptr, &psh, 3, .float32);
        defer _ = mlx.mlx_array_free(nrm);
        const qsh = [_]c_int{@intCast(q_abs.len)};
        const qidx = mlx.mlx_array_new_data(q_abs.ptr, &qsh, 1, .int32);
        defer _ = mlx.mlx_array_free(qidx);

        // 5. decode
        const ids = if (opts.use_kv_cache)
            try self.generateTokensGreedyCached(alloc, pc, nrm, qidx, opts.cls, opts.max_new)
        else
            try self.generateTokensGreedy(alloc, pc, nrm, qidx, opts.cls, opts.max_new);
        defer alloc.free(ids);

        // 6. detokenize + inverse-normalize joints/tails back to caller coords
        const skel = try utok.detokenize(alloc, ids);
        for (skel.joints) |*j| j.* = xform.inverseJoint(j.*);
        for (skel.tails) |*t| t.* = xform.inverseJoint(t.*);
        return skel;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Tests — hermetic config parse, then env-gated cos oracles fed by
// tests/dump_unirig_fixtures.py (UNIRIG_*; mirrors the HY3D_* harness).
// ════════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "unirig skeleton: config parse validates model_type and derived dims" {
    const cfg = try parseConfigText(
        \\{"model_type":"unirig_skeleton","quant":"fp16","ar":{"num_hidden_layers":24}}
    );
    try testing.expectEqual(@as(u32, 24), cfg.ar_layers);
    try testing.expectEqual(@as(u32, 64), cfg.arHeadDim());
    try testing.expectEqual(@as(u32, 64), cfg.encHeadDim());
    try testing.expectEqual(@as(u16, utok.Tok.cls_articulationxl), cfg.default_cls);
    try testing.expectError(error.BadUnirigConfig, parseConfigText(
        \\{"model_type":"hunyuan3d_2_1"}
    ));
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

fn readI64(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]i64 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(a, .limited(1024 * 1024 * 1024));
    defer a.free(bytes);
    const n = bytes.len / 8;
    const out = try a.alloc(i64, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 8]);
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
fn arrayCosine(arr: mlx.mlx_array, ref: []const f32, s: S) !f64 {
    const f = try astype(arr, .float32, s);
    defer _ = mlx.mlx_array_free(f);
    _ = mlx.mlx_array_eval(f);
    const n: usize = @intCast(mlx.mlx_array_size(f));
    try testing.expectEqual(ref.len, n);
    const d = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return cosine(d[0..n], ref);
}

/// Load the shared oracle inputs: pc [1,N,3], normals [1,N,3], q_idx [1024] int32.
const OracleIn = struct {
    pc_data: []f32,
    no_data: []f32,
    qidx_i32: []i32,
    pc: mlx.mlx_array,
    normals: mlx.mlx_array,
    qidx: mlx.mlx_array,
    n: usize,
    fn deinit(self: *OracleIn, a: std.mem.Allocator) void {
        _ = mlx.mlx_array_free(self.pc);
        _ = mlx.mlx_array_free(self.normals);
        _ = mlx.mlx_array_free(self.qidx);
        a.free(self.pc_data);
        a.free(self.no_data);
        a.free(self.qidx_i32);
    }
};

fn loadOracleIn(io: std.Io, a: std.mem.Allocator) !OracleIn {
    const pc_p = std.mem.span(std.c.getenv("UNIRIG_PC") orelse return error.SkipZigTest);
    const no_p = std.mem.span(std.c.getenv("UNIRIG_NORMALS") orelse return error.SkipZigTest);
    const qi_p = std.mem.span(std.c.getenv("UNIRIG_QIDX") orelse return error.SkipZigTest);
    const pc_data = try readF32(io, a, pc_p);
    errdefer a.free(pc_data);
    const no_data = try readF32(io, a, no_p);
    errdefer a.free(no_data);
    const qidx_i64 = try readI64(io, a, qi_p);
    defer a.free(qidx_i64);
    const n = pc_data.len / 3;
    const qidx_i32 = try a.alloc(i32, qidx_i64.len);
    errdefer a.free(qidx_i32);
    for (qidx_i64, 0..) |v, i| qidx_i32[i] = @intCast(v);
    const psh = [_]c_int{ 1, @intCast(n), 3 };
    const qsh = [_]c_int{@intCast(qidx_i32.len)};
    return .{
        .pc_data = pc_data,
        .no_data = no_data,
        .qidx_i32 = qidx_i32,
        .pc = mlx.mlx_array_new_data(pc_data.ptr, &psh, 3, .float32),
        .normals = mlx.mlx_array_new_data(no_data.ptr, &psh, 3, .float32),
        .qidx = mlx.mlx_array_new_data(qidx_i32.ptr, &qsh, 1, .int32),
        .n = n,
    };
}

// Oracle 1: michelangelo encoder latents [1,1024,512].
test "unirig oracle: encoder latents match reference" {
    const model_dir = std.mem.span(std.c.getenv("UNIRIG_TEST_MODEL") orelse return error.SkipZigTest);
    const enc_p = std.mem.span(std.c.getenv("UNIRIG_ENC") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var in = try loadOracleIn(io, a);
    defer in.deinit(a);
    const ref = try readF32(io, a, enc_p);
    defer a.free(ref);
    var eng = try Engine.load(io, a, model_dir);
    defer eng.deinit();
    const latents = try eng.encoder.encodeLatents(in.pc, in.normals, in.qidx);
    defer _ = mlx.mlx_array_free(latents);
    const corr = try arrayCosine(latents, ref, eng.s);
    std.debug.print("[unirig-enc] corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.99);
}

// Oracle 2: prefix embeds [1,1026,1024] (output_proj(latents) ++ embed[bos,cls]).
test "unirig oracle: prefix assembly matches reference" {
    const model_dir = std.mem.span(std.c.getenv("UNIRIG_TEST_MODEL") orelse return error.SkipZigTest);
    const pfx_p = std.mem.span(std.c.getenv("UNIRIG_PREFIX") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var in = try loadOracleIn(io, a);
    defer in.deinit(a);
    const ref = try readF32(io, a, pfx_p);
    defer a.free(ref);
    var eng = try Engine.load(io, a, model_dir);
    defer eng.deinit();
    const latents = try eng.encoder.encodeLatents(in.pc, in.normals, in.qidx);
    defer _ = mlx.mlx_array_free(latents);
    const prefix = try eng.buildPrefix(latents, eng.cfg.default_cls);
    defer _ = mlx.mlx_array_free(prefix);
    const corr = try arrayCosine(prefix, ref, eng.s);
    std.debug.print("[unirig-prefix] corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.99);
}

// Oracle 3: one OPT forward → next-token logits [267] at the last prefix position.
test "unirig oracle: first-step logits match reference" {
    const model_dir = std.mem.span(std.c.getenv("UNIRIG_TEST_MODEL") orelse return error.SkipZigTest);
    const sl_p = std.mem.span(std.c.getenv("UNIRIG_STEP_LOGITS") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var in = try loadOracleIn(io, a);
    defer in.deinit(a);
    const ref = try readF32(io, a, sl_p);
    defer a.free(ref);
    var eng = try Engine.load(io, a, model_dir);
    defer eng.deinit();
    const latents = try eng.encoder.encodeLatents(in.pc, in.normals, in.qidx);
    defer _ = mlx.mlx_array_free(latents);
    const prefix = try eng.buildPrefix(latents, eng.cfg.default_cls);
    defer _ = mlx.mlx_array_free(prefix);
    const logits = try eng.decoder.forward(prefix, eng.s); // [1,1026,267]
    defer _ = mlx.mlx_array_free(logits);
    const L = mlx.getShape(logits)[1];
    const last = try sliceAxis(logits, 1, L - 1, L, eng.s); // [1,1,267]
    defer _ = mlx.mlx_array_free(last);
    const corr = try arrayCosine(last, ref, eng.s);
    std.debug.print("[unirig-step] corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.99);
}

// Oracle 4: full greedy grammar-masked decode → token sequence. fp16 vs fp32-CPU
// argmax ties may diverge in the coordinate tail; require a valid skeleton + a
// matching structural/leading prefix (compared against the reference sequence).
test "unirig oracle: greedy decode reproduces the reference token prefix" {
    const model_dir = std.mem.span(std.c.getenv("UNIRIG_TEST_MODEL") orelse return error.SkipZigTest);
    const e2e_p = std.mem.span(std.c.getenv("UNIRIG_E2E_TOKENS") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var in = try loadOracleIn(io, a);
    defer in.deinit(a);
    const ref_i64 = try readI64(io, a, e2e_p);
    defer a.free(ref_i64);
    var eng = try Engine.load(io, a, model_dir);
    defer eng.deinit();
    const got = try eng.generateTokensGreedy(a, in.pc, in.normals, in.qidx, eng.cfg.default_cls, ref_i64.len + 8);
    defer a.free(got);

    // print both for calibration
    std.debug.print("[unirig-e2e] ref({d}): ", .{ref_i64.len});
    for (ref_i64) |t| std.debug.print("{d} ", .{t});
    std.debug.print("\n[unirig-e2e] got({d}): ", .{got.len});
    for (got) |t| std.debug.print("{d} ", .{t});
    std.debug.print("\n", .{});

    // must be a valid, detokenizable skeleton
    var skel = try utok.detokenize(a, got);
    defer skel.deinit(a);
    try testing.expect(skel.joints.len >= 1);

    // count the matching leading prefix vs the reference
    var matched: usize = 0;
    const lim = @min(got.len, ref_i64.len);
    while (matched < lim and @as(i64, got[matched]) == ref_i64[matched]) matched += 1;
    std.debug.print("[unirig-e2e] matched leading tokens = {d}/{d}\n", .{ matched, ref_i64.len });
    // bos + cls + spring + at least the first joint triple must agree exactly
    try testing.expect(matched >= 6);
}

// ── hermetic pure tests (no MLX, no weights) ──────────────────────────────────

test "unirig skeleton: NormXform normalizes into the cube and inverse round-trips" {
    var pts = [_]f32{ 1, 2, 3, 5, 2, 3, 1, 8, 3, 1, 2, 9 }; // off-center anisotropic box
    const orig = pts;
    const x = NormXform.compute(&pts);
    x.applyInPlace(&pts);
    for (pts) |c| try testing.expect(c >= -1.0001 and c <= 1.0001);
    var i: usize = 0;
    while (i < pts.len) : (i += 3) {
        const j = x.inverseJoint(.{ pts[i], pts[i + 1], pts[i + 2] });
        try testing.expectApproxEqAbs(orig[i], j[0], 1e-4);
        try testing.expectApproxEqAbs(orig[i + 1], j[1], 1e-4);
        try testing.expectApproxEqAbs(orig[i + 2], j[2], 1e-4);
    }
}

test "unirig skeleton: surface sampling lands on the mesh with unit normals" {
    const a = testing.allocator;
    const pos = [_]f32{ 0, 0, 0, 1, 0, 0, 0, 1, 0 }; // one triangle in z=0
    const nrm = [_]f32{ 0, 0, 1, 0, 0, 1, 0, 0, 1 };
    const faces = [_]u32{ 0, 1, 2 };
    const sm = try sampleSurface(a, &pos, &nrm, &faces, 256, 7);
    defer a.free(sm.pts);
    defer a.free(sm.nrm);
    var k: usize = 0;
    while (k < 256) : (k += 1) {
        try testing.expectApproxEqAbs(@as(f32, 0), sm.pts[k * 3 + 2], 1e-5); // on the plane
        try testing.expect(sm.pts[k * 3] >= -1e-5 and sm.pts[k * 3 + 1] >= -1e-5);
        try testing.expect(sm.pts[k * 3] + sm.pts[k * 3 + 1] <= 1.0001); // inside the triangle
        try testing.expectApproxEqAbs(@as(f32, 1), sm.nrm[k * 3 + 2], 1e-4); // unit +z
    }
}

test "unirig skeleton: presample returns distinct in-range indices" {
    const a = testing.allocator;
    const idx = try presampleIndices(a, 100, 40, 3);
    defer a.free(idx);
    try testing.expectEqual(@as(usize, 40), idx.len);
    for (idx) |v| try testing.expect(v < 100);
    for (0..idx.len) |i| for (i + 1..idx.len) |j| try testing.expect(idx[i] != idx[j]);
}

// ── hermetic KV-cache equivalence (tiny random decoder; no weights) ───────────

fn tinyDense(a: std.mem.Allocator, in: usize, out: usize, seed: u64, s: S) !MixedLinear {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const wd = try a.alloc(f32, in * out); // [in,out] pre-transposed (MixedLinear dense layout)
    defer a.free(wd);
    for (wd) |*d| d.* = (rng.float(f32) - 0.5) * 0.2;
    const wsh = [_]c_int{ @intCast(in), @intCast(out) };
    const wsrc = mlx.mlx_array_new_data(wd.ptr, &wsh, 2, .float32);
    defer _ = mlx.mlx_array_free(wsrc);
    const w = try astype(wsrc, .float16, s);
    _ = mlx.mlx_array_eval(w); // materialize before wd is freed
    const bd = try a.alloc(f32, out);
    defer a.free(bd);
    for (bd) |*d| d.* = (rng.float(f32) - 0.5) * 0.1;
    const bsh = [_]c_int{@intCast(out)};
    const bsrc = mlx.mlx_array_new_data(bd.ptr, &bsh, 1, .float32);
    defer _ = mlx.mlx_array_free(bsrc);
    const b = try astype(bsrc, .float16, s);
    _ = mlx.mlx_array_eval(b);
    return .{ .quantized = false, .w = w, .add_bias = b };
}

fn tinyVec(a: std.mem.Allocator, dim: usize, seed: u64, base: f32, s: S) !mlx.mlx_array {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const d = try a.alloc(f32, dim);
    defer a.free(d);
    for (d) |*x| x.* = base + (rng.float(f32) - 0.5) * 0.02;
    const sh = [_]c_int{@intCast(dim)};
    const src = mlx.mlx_array_new_data(d.ptr, &sh, 1, .float32);
    defer _ = mlx.mlx_array_free(src);
    const o = try astype(src, .float32, s);
    _ = mlx.mlx_array_eval(o);
    return o;
}

fn tinyEmbed(a: std.mem.Allocator, rows: usize, dim: usize, seed: u64, s: S) !mlx.mlx_array {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const d = try a.alloc(f32, rows * dim);
    defer a.free(d);
    for (d) |*x| x.* = (rng.float(f32) - 0.5) * 0.5;
    const sh = [_]c_int{ @intCast(rows), @intCast(dim) };
    const src = mlx.mlx_array_new_data(d.ptr, &sh, 2, .float32);
    defer _ = mlx.mlx_array_free(src);
    const o = try astype(src, .float16, s);
    _ = mlx.mlx_array_eval(o);
    return o;
}

fn buildTinyDecoder(a: std.mem.Allocator, s: S) !Decoder {
    const cfg = SkelConfig{ .ar_layers = 2, .ar_hidden = 32, .ar_heads = 4, .ar_ffn = 64, .pos_offset = 2, .vocab = 16 };
    var dec: Decoder = undefined;
    dec.cfg = cfg;
    dec.allocator = a;
    dec.s = s;
    dec.embed_tokens = try tinyEmbed(a, 16, 32, 1, s);
    dec.embed_positions = try tinyEmbed(a, 64, 32, 2, s);
    dec.layers = try a.alloc(OptLayer, cfg.ar_layers);
    for (dec.layers, 0..) |*l, i| {
        const si: u64 = @intCast(i);
        l.* = .{
            .attn_norm_w = try tinyVec(a, 32, 100 + si, 1.0, s),
            .attn_norm_b = try tinyVec(a, 32, 200 + si, 0.0, s),
            .q = try tinyDense(a, 32, 32, 300 + si, s),
            .k = try tinyDense(a, 32, 32, 400 + si, s),
            .v = try tinyDense(a, 32, 32, 500 + si, s),
            .o = try tinyDense(a, 32, 32, 600 + si, s),
            .mlp_norm_w = try tinyVec(a, 32, 700 + si, 1.0, s),
            .mlp_norm_b = try tinyVec(a, 32, 800 + si, 0.0, s),
            .fc1 = try tinyDense(a, 32, 64, 900 + si, s),
            .fc2 = try tinyDense(a, 64, 32, 1000 + si, s),
        };
    }
    dec.final_norm_w = try tinyVec(a, 32, 1100, 1.0, s);
    dec.final_norm_b = try tinyVec(a, 32, 1200, 0.0, s);
    dec.lm_head = try tinyDense(a, 32, 16, 1300, s);
    return dec;
}

fn argmaxLastIdx(logits: mlx.mlx_array, s: S) !usize {
    const L = mlx.getShape(logits)[1];
    const last = try sliceAxis(logits, 1, L - 1, L, s);
    defer _ = mlx.mlx_array_free(last);
    const lf = try astype(last, .float32, s);
    defer _ = mlx.mlx_array_free(lf);
    _ = mlx.mlx_array_eval(lf);
    const n: usize = @intCast(mlx.mlx_array_size(lf));
    const d = mlx.mlx_array_data_float32(lf) orelse return error.NoData;
    var bi: usize = 0;
    var bv: f32 = d[0];
    for (1..n) |i| if (d[i] > bv) {
        bv = d[i];
        bi = i;
    };
    return bi;
}

test "unirig skeleton: KV-cache decode == no-cache forward (hermetic tiny decoder)" {
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    var dec = try buildTinyDecoder(a, s);
    defer dec.deinit();
    const T: usize = 6;
    const H: usize = 32;
    const ed = try a.alloc(f32, T * H);
    defer a.free(ed);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    for (ed) |*x| x.* = (rng.float(f32) - 0.5) * 0.5;
    const esh = [_]c_int{ 1, @intCast(T), @intCast(H) };
    const esrc = mlx.mlx_array_new_data(ed.ptr, &esh, 3, .float32);
    defer _ = mlx.mlx_array_free(esrc);
    const embeds = try astype(esrc, .float16, s);
    defer _ = mlx.mlx_array_free(embeds);
    _ = mlx.mlx_array_eval(embeds);

    // cached: prefill token 0, then feed one token per step; argmax at each step
    var cache = try KvCache.init(a, dec.cfg.ar_layers);
    defer cache.deinit(a);
    var cached_arg: [T]usize = undefined;
    for (0..T) |t| {
        const step = try sliceAxis(embeds, 1, @intCast(t), @intCast(t + 1), s);
        defer _ = mlx.mlx_array_free(step);
        const lg = try dec.forwardCached(step, &cache, t, s);
        defer _ = mlx.mlx_array_free(lg);
        cached_arg[t] = try argmaxLastIdx(lg, s);
    }
    // no-cache: forward(embeds[0..t+1]); last-position argmax must match the cached step
    for (0..T) |t| {
        const pre = try sliceAxis(embeds, 1, 0, @intCast(t + 1), s);
        defer _ = mlx.mlx_array_free(pre);
        const lg = try dec.forward(pre, s);
        defer _ = mlx.mlx_array_free(lg);
        const nc = try argmaxLastIdx(lg, s);
        try testing.expectEqual(cached_arg[t], nc);
    }
}

// ── env-gated: 8-bit shipping build + cache/live paths on the real model ──────

test "unirig oracle 8-bit: shipping build stays faithful (enc/prefix/step/e2e)" {
    const model_dir = std.mem.span(std.c.getenv("UNIRIG_TEST_MODEL_8BIT") orelse return error.SkipZigTest);
    const enc_p = std.mem.span(std.c.getenv("UNIRIG_ENC") orelse return error.SkipZigTest);
    const pfx_p = std.mem.span(std.c.getenv("UNIRIG_PREFIX") orelse return error.SkipZigTest);
    const sl_p = std.mem.span(std.c.getenv("UNIRIG_STEP_LOGITS") orelse return error.SkipZigTest);
    const e2e_p = std.mem.span(std.c.getenv("UNIRIG_E2E_TOKENS") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var in = try loadOracleIn(io, a);
    defer in.deinit(a);
    const enc_ref = try readF32(io, a, enc_p);
    defer a.free(enc_ref);
    const pfx_ref = try readF32(io, a, pfx_p);
    defer a.free(pfx_ref);
    const step_ref = try readF32(io, a, sl_p);
    defer a.free(step_ref);
    const e2e_ref = try readI64(io, a, e2e_p);
    defer a.free(e2e_ref);
    var eng = try Engine.load(io, a, model_dir);
    defer eng.deinit();
    // 8-bit affine (gs64) — honest headroom below the fp16 build's ~0.99999
    const latents = try eng.encoder.encodeLatents(in.pc, in.normals, in.qidx);
    defer _ = mlx.mlx_array_free(latents);
    const c_enc = try arrayCosine(latents, enc_ref, eng.s);
    const prefix = try eng.buildPrefix(latents, eng.cfg.default_cls);
    defer _ = mlx.mlx_array_free(prefix);
    const c_pfx = try arrayCosine(prefix, pfx_ref, eng.s);
    const logits = try eng.decoder.forward(prefix, eng.s);
    defer _ = mlx.mlx_array_free(logits);
    const Lc = mlx.getShape(logits)[1];
    const last = try sliceAxis(logits, 1, Lc - 1, Lc, eng.s);
    defer _ = mlx.mlx_array_free(last);
    const c_step = try arrayCosine(last, step_ref, eng.s);
    std.debug.print("[unirig-8bit] enc={d:.6} prefix={d:.6} step={d:.6}\n", .{ c_enc, c_pfx, c_step });
    try testing.expect(c_enc > 0.997 and c_pfx > 0.997 and c_step > 0.997);
    // e2e: valid grammar + a matching leading prefix (8-bit may drift in the tail)
    const got = try eng.generateTokensGreedy(a, in.pc, in.normals, in.qidx, eng.cfg.default_cls, e2e_ref.len + 8);
    defer a.free(got);
    var skel = try utok.detokenize(a, got);
    defer skel.deinit(a);
    var matched: usize = 0;
    const lim = @min(got.len, e2e_ref.len);
    while (matched < lim and @as(i64, got[matched]) == e2e_ref[matched]) matched += 1;
    std.debug.print("[unirig-8bit] e2e matched {d}/{d}\n", .{ matched, e2e_ref.len });
    try testing.expect(matched >= 6);
}

test "unirig oracle: KV-cache greedy == no-cache greedy on the real model" {
    const model_dir = std.mem.span(std.c.getenv("UNIRIG_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var in = try loadOracleIn(io, a);
    defer in.deinit(a);
    var eng = try Engine.load(io, a, model_dir);
    defer eng.deinit();
    const nc = try eng.generateTokensGreedy(a, in.pc, in.normals, in.qidx, eng.cfg.default_cls, 64);
    defer a.free(nc);
    const cc = try eng.generateTokensGreedyCached(a, in.pc, in.normals, in.qidx, eng.cfg.default_cls, 64);
    defer a.free(cc);
    std.debug.print("[unirig-kv] no-cache={d} toks, cached={d} toks\n", .{ nc.len, cc.len });
    try testing.expectEqual(nc.len, cc.len);
    try testing.expectEqualSlices(u16, nc, cc);
}

test "unirig oracle: generateSkeleton live path returns a valid skeleton in caller coords" {
    const model_dir = std.mem.span(std.c.getenv("UNIRIG_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var eng = try Engine.load(io, a, model_dir);
    defer eng.deinit();
    // a unit cube mesh, offset to a non-cube-centered frame, to exercise
    // normalize → sample → decode → inverse-normalize.
    const off = [3]f32{ 10.0, -5.0, 2.0 };
    var verts: [8 * 3]f32 = undefined;
    const corners = [8][3]f32{
        .{ 0, 0, 0 }, .{ 1, 0, 0 }, .{ 1, 1, 0 }, .{ 0, 1, 0 },
        .{ 0, 0, 1 }, .{ 1, 0, 1 }, .{ 1, 1, 1 }, .{ 0, 1, 1 },
    };
    for (corners, 0..) |cnr, i| for (0..3) |c| {
        verts[i * 3 + c] = cnr[c] + off[c];
    };
    // crude per-vertex normals (radial from cube center) — fine for a smoke test
    var norms: [8 * 3]f32 = undefined;
    for (0..8) |i| {
        const v = [3]f32{ verts[i * 3] - (off[0] + 0.5), verts[i * 3 + 1] - (off[1] + 0.5), verts[i * 3 + 2] - (off[2] + 0.5) };
        const l = @sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        for (0..3) |c| norms[i * 3 + c] = v[c] / l;
    }
    const faces = [_]u32{
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1,
        1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0,
    };
    var skel = try eng.generateSkeleton(a, &verts, &norms, &faces, .{ .num_samples = 4096, .max_new = 128 });
    defer skel.deinit(a);
    std.debug.print("[unirig-live] {d} joints\n", .{skel.joints.len});
    try testing.expect(skel.joints.len >= 1);
    try testing.expectEqual(skel.joints.len, skel.parents.len);
    try testing.expectEqual(skel.joints.len, skel.tails.len);
    // joints came back in the caller's frame (finite, near the cube's world range)
    for (skel.joints) |j| for (j) |c| try testing.expect(std.math.isFinite(c));
    try testing.expectEqual(@as(?usize, null), skel.parents[0]); // root
}
