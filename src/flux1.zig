//! Native FLUX.1-dev / FLUX.1-schnell text→image (T5-XXL + CLIP-L text encoders,
//! MMDiT with per-block modulation, FlowMatchEuler sampler, 16-ch VAE). Ported to
//! mlx-c FFI from the diffusers FluxTransformer2DModel reference. The T5/CLIP
//! encoders live in t5.zig/flux1_clip.zig, the VAE in flux1_vae.zig. See
//! docs/reference.md (FLUX.1 section).
//!
//! dev vs schnell is one weight: `guidance_embedder` present ⇒ dev (guidance
//! embed + default 3.5 guidance + dynamic timestep shift); absent ⇒ schnell
//! (no guidance, 4 steps, no shift). Everything else is shared.
//!
//! Transformer keys are diffusers-style (`transformer_blocks.N.*`,
//! `single_transformer_blocks.N.*`, `time_text_embed.*`); linears affine quantized.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const sse = @import("gen_sse.zig");
const fc = @import("flux_common.zig");
const lora_mod = @import("lora.zig");
const t5 = @import("t5.zig");
const clip_mod = @import("flux1_clip.zig");
const vae_mod = @import("flux1_vae.zig");
const t5tok = @import("t5_tokenizer.zig");

const QLinear = fc.QLinear;
const S = fc.S;
const Weights = model_mod.Weights;
const addA = fc.addA;
const mulA = fc.mulA;
const subA = fc.subA;
const reshape = fc.reshape;
const transpose = fc.transpose;
const astype = fc.astype;
const silu = fc.silu;
const geluTanh = fc.geluTanh;
const concat = fc.concat;
const slice3 = fc.slice3;
const rms = fc.rms;

const T5_MAX_LEN: usize = 512;
const CLIP_MAX_LEN: usize = 77;
const AXES_DIM = [3]u32{ 16, 56, 56 }; // FLUX rope: sums to head_dim 128
const ROPE_THETA: f64 = 10000.0;

pub const Flux1Config = struct {
    inner: u32 = 3072,
    heads: u32 = 24,
    head_dim: u32 = 128,
    double_layers: u32 = 19,
    single_layers: u32 = 38,
    joint_dim: u32 = 4096, // T5 context width
    pooled_dim: u32 = 768, // CLIP pooled width
    x_in_dim: u32 = 64,
    guidance: bool = true, // dev
};

// ── generic block helpers ──

inline fn meanLast(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_mean_axis(&o, x, -1, true, s));
    return o;
}
/// LayerNorm, affine=false: (x-mean)/sqrt(var+eps) over the last dim.
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
    var rsv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rsv);
    try mlx.check(mlx.mlx_rsqrt(&rsv, ve, s));
    const out = try mulA(xc, rsv, s);
    defer _ = mlx.mlx_array_free(out);
    return astype(out, .bfloat16, s);
}
/// (1+scale)*ln + shift.
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
/// Split modulation [1,1,n*inner] into n chunks of [1,1,inner].
fn modChunks(mod: mlx.mlx_array, inner: c_int, n: usize, s: S) ![6]mlx.mlx_array {
    var out: [6]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null }, .{ .ctx = null }, .{ .ctx = null }, .{ .ctx = null }, .{ .ctx = null } };
    for (0..n) |i| {
        out[i] = try slice3(mod, 2, @as(c_int, @intCast(i)) * inner, @as(c_int, @intCast(i + 1)) * inner, s);
    }
    return out;
}
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

const DoubleBlock = struct {
    norm1_lin: QLinear, // img AdaLayerNormZero
    norm1c_lin: QLinear, // txt AdaLayerNormZero
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
    ff1: QLinear,
    ff2: QLinear,
    ffc1: QLinear,
    ffc2: QLinear,
    fn deinit(self: *DoubleBlock) void {
        inline for (.{ "norm1_lin", "norm1c_lin", "q", "k", "v", "o", "add_q", "add_k", "add_v", "add_o", "ff1", "ff2", "ffc1", "ffc2" }) |f| @field(self, f).deinit();
        inline for (.{ "nq", "nk", "naq", "nak" }) |f| _ = mlx.mlx_array_free(@field(self, f));
    }
};
const SingleBlock = struct {
    norm_lin: QLinear,
    q: QLinear,
    k: QLinear,
    v: QLinear,
    nq: mlx.mlx_array,
    nk: mlx.mlx_array,
    proj_mlp: QLinear,
    proj_out: QLinear,
    fn deinit(self: *SingleBlock) void {
        inline for (.{ "norm_lin", "q", "k", "v", "proj_mlp", "proj_out" }) |f| @field(self, f).deinit();
        _ = mlx.mlx_array_free(self.nq);
        _ = mlx.mlx_array_free(self.nk);
    }
};

pub const Dit = struct {
    cfg: Flux1Config,
    allocator: std.mem.Allocator,
    s: S,
    x_embedder: QLinear,
    context_embedder: QLinear,
    ts_lin1: QLinear,
    ts_lin2: QLinear,
    guid_lin1: ?QLinear,
    guid_lin2: ?QLinear,
    txt_lin1: QLinear,
    txt_lin2: QLinear,
    doubles: []DoubleBlock,
    singles: []SingleBlock,
    norm_out_lin: QLinear,
    proj_out: QLinear,

    pub fn deinit(self: *Dit) void {
        inline for (.{ "x_embedder", "context_embedder", "ts_lin1", "ts_lin2", "txt_lin1", "txt_lin2", "norm_out_lin", "proj_out" }) |f| @field(self, f).deinit();
        if (self.guid_lin1) |*g| g.deinit();
        if (self.guid_lin2) |*g| g.deinit();
        for (self.doubles) |*b| b.deinit();
        self.allocator.free(self.doubles);
        for (self.singles) |*b| b.deinit();
        self.allocator.free(self.singles);
    }

    /// Sinusoidal timestep embedding (dim 256, flip_sin_to_cos, max_period 1e4).
    fn sinEmbed(self: *Dit, t: f32) !mlx.mlx_array {
        const s = self.s;
        var buf: [256]f32 = undefined;
        const half = 128;
        for (0..half) |i| {
            const freq = std.math.exp(-std.math.log(f32, std.math.e, 10000.0) * @as(f32, @floatFromInt(i)) / @as(f32, half));
            const arg = t * freq;
            buf[half + i] = @sin(arg);
            buf[i] = @cos(arg); // flip_sin_to_cos: [cos..., sin...]
        }
        const shape = [_]c_int{ 1, 256 };
        const te = mlx.mlx_array_new_data(&buf, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(te);
        return astype(te, .bfloat16, s);
    }

    /// temb [1,1,inner] = timestep_emb + (guidance_emb if dev) + text_emb(pooled).
    /// `t` = sigma·1000, `guid` = guidance_scale·1000, pooled = CLIP [1,1,768].
    fn conditioning(self: *Dit, t: f32, guid: f32, pooled: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const inner: c_int = @intCast(self.cfg.inner);
        // timestep
        const tse = try self.sinEmbed(t);
        defer _ = mlx.mlx_array_free(tse);
        const tl1 = try self.ts_lin1.forward(tse, s);
        defer _ = mlx.mlx_array_free(tl1);
        const ta = try silu(tl1, s);
        defer _ = mlx.mlx_array_free(ta);
        var temb = try self.ts_lin2.forward(ta, s); // [1,inner]
        // + guidance
        if (self.cfg.guidance) {
            const gse = try self.sinEmbed(guid);
            defer _ = mlx.mlx_array_free(gse);
            const gl1 = try self.guid_lin1.?.forward(gse, s);
            defer _ = mlx.mlx_array_free(gl1);
            const ga = try silu(gl1, s);
            defer _ = mlx.mlx_array_free(ga);
            const gemb = try self.guid_lin2.?.forward(ga, s);
            defer _ = mlx.mlx_array_free(gemb);
            const nt = try addA(temb, gemb, s);
            _ = mlx.mlx_array_free(temb);
            temb = nt;
        }
        // + text (pooled CLIP)
        {
            const pl1 = try self.txt_lin1.forward(pooled, s); // pooled [1,1,768] → [1,1,inner]
            defer _ = mlx.mlx_array_free(pl1);
            const pa = try silu(pl1, s);
            defer _ = mlx.mlx_array_free(pa);
            const pemb = try self.txt_lin2.forward(pa, s); // [1,1,inner]
            defer _ = mlx.mlx_array_free(pemb);
            const p2 = try reshape(pemb, &[_]c_int{ 1, inner }, s);
            defer _ = mlx.mlx_array_free(p2);
            const nt = try addA(temb, p2, s);
            _ = mlx.mlx_array_free(temb);
            temb = nt;
        }
        const out = try reshape(temb, &[_]c_int{ 1, 1, inner }, s);
        _ = mlx.mlx_array_free(temb);
        return out;
    }

    /// Interleaved-RoPE cos/sin [seq,64] from ids [seq,3] with axes_dim [16,56,56].
    fn buildRope(self: *Dit, ids: []const i32, seq: usize) !struct { cos: mlx.mlx_array, sin: mlx.mlx_array } {
        const cosb = try self.allocator.alloc(f32, seq * 64);
        defer self.allocator.free(cosb);
        const sinb = try self.allocator.alloc(f32, seq * 64);
        defer self.allocator.free(sinb);
        var pair_off: [3]usize = undefined;
        var acc: usize = 0;
        for (0..3) |ax| {
            pair_off[ax] = acc;
            acc += AXES_DIM[ax] / 2;
        }
        for (0..seq) |p| {
            for (0..3) |ax| {
                const dim = AXES_DIM[ax];
                const pos: f64 = @floatFromInt(ids[p * 3 + ax]);
                const npairs = dim / 2;
                for (0..npairs) |kk| {
                    const omega = std.math.pow(f64, ROPE_THETA, -@as(f64, @floatFromInt(2 * kk)) / @as(f64, @floatFromInt(dim)));
                    const ang = pos * omega;
                    const idx = p * 64 + pair_off[ax] + kk;
                    cosb[idx] = @floatCast(@cos(ang));
                    sinb[idx] = @floatCast(@sin(ang));
                }
            }
        }
        const shape = [_]c_int{ @intCast(seq), 64 };
        return .{
            .cos = mlx.mlx_array_new_data(cosb.ptr, &shape, 2, .float32),
            .sin = mlx.mlx_array_new_data(sinb.ptr, &shape, 2, .float32),
        };
    }

    /// Apply interleaved RoPE to q [1,heads,seq,128] using cos/sin [seq,64].
    fn applyRope(self: *Dit, q: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, seq: c_int, heads: c_int) !mlx.mlx_array {
        const s = self.s;
        const qf = try astype(q, .float32, s);
        defer _ = mlx.mlx_array_free(qf);
        const q5 = try reshape(qf, &[_]c_int{ 1, heads, seq, 64, 2 }, s);
        defer _ = mlx.mlx_array_free(q5);
        const real = try sliceLast2(q5, 0, s);
        defer _ = mlx.mlx_array_free(real);
        const imag = try sliceLast2(q5, 1, s);
        defer _ = mlx.mlx_array_free(imag);
        const cos_b = try reshape(cos, &[_]c_int{ 1, 1, seq, 64 }, s);
        defer _ = mlx.mlx_array_free(cos_b);
        const sin_b = try reshape(sin, &[_]c_int{ 1, 1, seq, 64 }, s);
        defer _ = mlx.mlx_array_free(sin_b);
        const rc = try mulA(real, cos_b, s);
        defer _ = mlx.mlx_array_free(rc);
        const is_ = try mulA(imag, sin_b, s);
        defer _ = mlx.mlx_array_free(is_);
        const out0 = try subA(rc, is_, s);
        defer _ = mlx.mlx_array_free(out0);
        const ic = try mulA(imag, cos_b, s);
        defer _ = mlx.mlx_array_free(ic);
        const rs2 = try mulA(real, sin_b, s);
        defer _ = mlx.mlx_array_free(rs2);
        const out1 = try addA(ic, rs2, s);
        defer _ = mlx.mlx_array_free(out1);
        const o0e = try reshape(out0, &[_]c_int{ 1, heads, seq, 64, 1 }, s);
        defer _ = mlx.mlx_array_free(o0e);
        const o1e = try reshape(out1, &[_]c_int{ 1, heads, seq, 64, 1 }, s);
        defer _ = mlx.mlx_array_free(o1e);
        const st = try concat(&[_]mlx.mlx_array{ o0e, o1e }, 4, s);
        defer _ = mlx.mlx_array_free(st);
        const flat = try reshape(st, &[_]c_int{ 1, heads, seq, 128 }, s);
        defer _ = mlx.mlx_array_free(flat);
        return astype(flat, .bfloat16, s);
    }

    fn splitHeads(self: *Dit, x: mlx.mlx_array, seq: c_int, heads: c_int, hd: c_int, norm: ?mlx.mlx_array, s: S) !mlx.mlx_array {
        _ = self;
        const x4 = try reshape(x, &[_]c_int{ 1, seq, heads, hd }, s);
        defer _ = mlx.mlx_array_free(x4);
        const xt = try transpose(x4, &[_]c_int{ 0, 2, 1, 3 }, s);
        if (norm) |nw| {
            defer _ = mlx.mlx_array_free(xt);
            const xf = try astype(xt, .float32, s);
            defer _ = mlx.mlx_array_free(xf);
            const xn = try rms(xf, nw, 1e-6, s);
            defer _ = mlx.mlx_array_free(xn);
            return astype(xn, .bfloat16, s);
        }
        return xt;
    }

    fn attention(self: *Dit, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, seq: c_int, heads: c_int, hd: c_int, s: S) !mlx.mlx_array {
        _ = self;
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(hd)));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_a = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, q, k, v, scale, "", null_a, null_a, false, s));
        const at = try transpose(attn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        return reshape(at, &[_]c_int{ 1, seq, heads * hd }, s);
    }

    /// Plain FF: linear1 → gelu(tanh) → linear2.
    fn feedForward(self: *Dit, x: mlx.mlx_array, l1: *const QLinear, l2: *const QLinear, s: S) !mlx.mlx_array {
        _ = self;
        const h = try l1.forward(x, s);
        defer _ = mlx.mlx_array_free(h);
        const g = try geluTanh(h, s);
        defer _ = mlx.mlx_array_free(g);
        return l2.forward(g, s);
    }

    pub fn forward(self: *Dit, latents: mlx.mlx_array, context: mlx.mlx_array, pooled: mlx.mlx_array, t: f32, guid: f32, img_ids: []const i32, txt_ids: []const i32) !mlx.mlx_array {
        const s = self.s;
        const c = self.cfg;
        const inner: c_int = @intCast(c.inner);
        const heads: c_int = @intCast(c.heads);
        const hd: c_int = @intCast(c.head_dim);
        const Nt: c_int = mlx.getShape(context)[1];
        const Ni: c_int = mlx.getShape(latents)[1];
        const Nj = Nt + Ni;

        const temb = try self.conditioning(t, guid, pooled);
        defer _ = mlx.mlx_array_free(temb);
        const stemb = try silu(temb, s);
        defer _ = mlx.mlx_array_free(stemb);

        var hs = try self.x_embedder.forward(latents, s);
        var ehs = try self.context_embedder.forward(context, s);

        const tr = try self.buildRope(txt_ids, @intCast(Nt));
        defer {
            _ = mlx.mlx_array_free(tr.cos);
            _ = mlx.mlx_array_free(tr.sin);
        }
        const ir = try self.buildRope(img_ids, @intCast(Ni));
        defer {
            _ = mlx.mlx_array_free(ir.cos);
            _ = mlx.mlx_array_free(ir.sin);
        }
        const cos = try concat(&[_]mlx.mlx_array{ tr.cos, ir.cos }, 0, s);
        defer _ = mlx.mlx_array_free(cos);
        const sin = try concat(&[_]mlx.mlx_array{ tr.sin, ir.sin }, 0, s);
        defer _ = mlx.mlx_array_free(sin);

        for (self.doubles) |*b| {
            const r = try self.doubleBlock(hs, ehs, b, stemb, cos, sin, Nt, Ni, heads, hd, inner, s);
            _ = mlx.mlx_array_free(hs);
            _ = mlx.mlx_array_free(ehs);
            hs = r.img;
            ehs = r.txt;
        }
        var joint = try concat(&[_]mlx.mlx_array{ ehs, hs }, 1, s);
        _ = mlx.mlx_array_free(hs);
        _ = mlx.mlx_array_free(ehs);
        for (self.singles) |*b| {
            const nj = try self.singleBlock(joint, b, stemb, cos, sin, Nj, heads, hd, inner, s);
            _ = mlx.mlx_array_free(joint);
            joint = nj;
        }
        const img = try slice3(joint, 1, Nt, Nj, s);
        _ = mlx.mlx_array_free(joint);
        defer _ = mlx.mlx_array_free(img);
        // norm_out (AdaLayerNormContinuous): linear(silu(temb)) → [scale, shift]
        const so = try self.norm_out_lin.forward(stemb, s);
        defer _ = mlx.mlx_array_free(so);
        const scale = try slice3(so, 2, 0, inner, s);
        defer _ = mlx.mlx_array_free(scale);
        const shift = try slice3(so, 2, inner, 2 * inner, s);
        defer _ = mlx.mlx_array_free(shift);
        const ln = try layerNormNA(img, 1e-6, s);
        defer _ = mlx.mlx_array_free(ln);
        const modded = try modulate(ln, scale, shift, s);
        defer _ = mlx.mlx_array_free(modded);
        return self.proj_out.forward(modded, s); // [1,Ni,64]
    }

    fn doubleBlock(self: *Dit, hs: mlx.mlx_array, ehs: mlx.mlx_array, b: *const DoubleBlock, stemb: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, Nt: c_int, Ni: c_int, heads: c_int, hd: c_int, inner: c_int, s: S) !struct { img: mlx.mlx_array, txt: mlx.mlx_array } {
        const mi_raw = try b.norm1_lin.forward(stemb, s);
        defer _ = mlx.mlx_array_free(mi_raw);
        const mt_raw = try b.norm1c_lin.forward(stemb, s);
        defer _ = mlx.mlx_array_free(mt_raw);
        const im = try modChunks(mi_raw, inner, 6, s);
        defer for (im) |x| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x);
        };
        const tm = try modChunks(mt_raw, inner, 6, s);
        defer for (tm) |x| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x);
        };
        const nh = try layerNormNA(hs, 1e-6, s);
        defer _ = mlx.mlx_array_free(nh);
        const nhm = try modulate(nh, im[1], im[0], s);
        defer _ = mlx.mlx_array_free(nhm);
        const ne = try layerNormNA(ehs, 1e-6, s);
        defer _ = mlx.mlx_array_free(ne);
        const nem = try modulate(ne, tm[1], tm[0], s);
        defer _ = mlx.mlx_array_free(nem);
        // image qkv
        const q = try b.q.forward(nhm, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try b.k.forward(nhm, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try b.v.forward(nhm, s);
        defer _ = mlx.mlx_array_free(v);
        const qh = try self.splitHeads(q, Ni, heads, hd, b.nq, s);
        defer _ = mlx.mlx_array_free(qh);
        const kh = try self.splitHeads(k, Ni, heads, hd, b.nk, s);
        defer _ = mlx.mlx_array_free(kh);
        const vh = try self.splitHeads(v, Ni, heads, hd, null, s);
        defer _ = mlx.mlx_array_free(vh);
        // text qkv
        const eq = try b.add_q.forward(nem, s);
        defer _ = mlx.mlx_array_free(eq);
        const ek = try b.add_k.forward(nem, s);
        defer _ = mlx.mlx_array_free(ek);
        const ev = try b.add_v.forward(nem, s);
        defer _ = mlx.mlx_array_free(ev);
        const eqh = try self.splitHeads(eq, Nt, heads, hd, b.naq, s);
        defer _ = mlx.mlx_array_free(eqh);
        const ekh = try self.splitHeads(ek, Nt, heads, hd, b.nak, s);
        defer _ = mlx.mlx_array_free(ekh);
        const evh = try self.splitHeads(ev, Nt, heads, hd, null, s);
        defer _ = mlx.mlx_array_free(evh);
        // concat [text, image] on seq (axis 2)
        const qj = try concat(&[_]mlx.mlx_array{ eqh, qh }, 2, s);
        defer _ = mlx.mlx_array_free(qj);
        const kj = try concat(&[_]mlx.mlx_array{ ekh, kh }, 2, s);
        defer _ = mlx.mlx_array_free(kj);
        const vj = try concat(&[_]mlx.mlx_array{ evh, vh }, 2, s);
        defer _ = mlx.mlx_array_free(vj);
        const Nj = Nt + Ni;
        const qr = try self.applyRope(qj, cos, sin, Nj, heads);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try self.applyRope(kj, cos, sin, Nj, heads);
        defer _ = mlx.mlx_array_free(kr);
        const attn = try self.attention(qr, kr, vj, Nj, heads, hd, s);
        defer _ = mlx.mlx_array_free(attn);
        const a_txt = try slice3(attn, 1, 0, Nt, s);
        defer _ = mlx.mlx_array_free(a_txt);
        const a_img = try slice3(attn, 1, Nt, Nj, s);
        defer _ = mlx.mlx_array_free(a_img);
        const ao_img = try b.o.forward(a_img, s);
        defer _ = mlx.mlx_array_free(ao_img);
        const ao_txt = try b.add_o.forward(a_txt, s);
        defer _ = mlx.mlx_array_free(ao_txt);
        const g_img = try mulA(im[2], ao_img, s);
        defer _ = mlx.mlx_array_free(g_img);
        const hs1 = try addA(hs, g_img, s);
        defer _ = mlx.mlx_array_free(hs1);
        const g_txt = try mulA(tm[2], ao_txt, s);
        defer _ = mlx.mlx_array_free(g_txt);
        const ehs1 = try addA(ehs, g_txt, s);
        defer _ = mlx.mlx_array_free(ehs1);
        // FF image
        const nh2 = try layerNormNA(hs1, 1e-6, s);
        defer _ = mlx.mlx_array_free(nh2);
        const nh2m = try modulate(nh2, im[4], im[3], s);
        defer _ = mlx.mlx_array_free(nh2m);
        const ff_img = try self.feedForward(nh2m, &b.ff1, &b.ff2, s);
        defer _ = mlx.mlx_array_free(ff_img);
        const gff_img = try mulA(im[5], ff_img, s);
        defer _ = mlx.mlx_array_free(gff_img);
        const img_out = try addA(hs1, gff_img, s);
        // FF text
        const ne2 = try layerNormNA(ehs1, 1e-6, s);
        defer _ = mlx.mlx_array_free(ne2);
        const ne2m = try modulate(ne2, tm[4], tm[3], s);
        defer _ = mlx.mlx_array_free(ne2m);
        const ff_txt = try self.feedForward(ne2m, &b.ffc1, &b.ffc2, s);
        defer _ = mlx.mlx_array_free(ff_txt);
        const gff_txt = try mulA(tm[5], ff_txt, s);
        defer _ = mlx.mlx_array_free(gff_txt);
        const txt_out = try addA(ehs1, gff_txt, s);
        return .{ .img = img_out, .txt = txt_out };
    }

    fn singleBlock(self: *Dit, hs: mlx.mlx_array, b: *const SingleBlock, stemb: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, Nj: c_int, heads: c_int, hd: c_int, inner: c_int, s: S) !mlx.mlx_array {
        const m_raw = try b.norm_lin.forward(stemb, s);
        defer _ = mlx.mlx_array_free(m_raw);
        const m = try modChunks(m_raw, inner, 3, s);
        defer for (m) |x| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x);
        };
        const nh = try layerNormNA(hs, 1e-6, s);
        defer _ = mlx.mlx_array_free(nh);
        const nhm = try modulate(nh, m[1], m[0], s);
        defer _ = mlx.mlx_array_free(nhm);
        const q = try b.q.forward(nhm, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try b.k.forward(nhm, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try b.v.forward(nhm, s);
        defer _ = mlx.mlx_array_free(v);
        const qh = try self.splitHeads(q, Nj, heads, hd, b.nq, s);
        defer _ = mlx.mlx_array_free(qh);
        const kh = try self.splitHeads(k, Nj, heads, hd, b.nk, s);
        defer _ = mlx.mlx_array_free(kh);
        const vh = try self.splitHeads(v, Nj, heads, hd, null, s);
        defer _ = mlx.mlx_array_free(vh);
        const qr = try self.applyRope(qh, cos, sin, Nj, heads);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try self.applyRope(kh, cos, sin, Nj, heads);
        defer _ = mlx.mlx_array_free(kr);
        const attn = try self.attention(qr, kr, vh, Nj, heads, hd, s);
        defer _ = mlx.mlx_array_free(attn);
        const mlp = try b.proj_mlp.forward(nhm, s);
        defer _ = mlx.mlx_array_free(mlp);
        const mlp_act = try geluTanh(mlp, s);
        defer _ = mlx.mlx_array_free(mlp_act);
        const cat = try concat(&[_]mlx.mlx_array{ attn, mlp_act }, 2, s);
        defer _ = mlx.mlx_array_free(cat);
        const ao = try b.proj_out.forward(cat, s);
        defer _ = mlx.mlx_array_free(ao);
        const g = try mulA(m[2], ao, s);
        defer _ = mlx.mlx_array_free(g);
        return addA(hs, g, s);
    }
};

// ── loaders ──

fn qkNorm(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, sub: []const u8) !mlx.mlx_array {
    const key = try std.fmt.allocPrint(a, "{s}.{s}", .{ pfx, sub });
    defer a.free(key);
    return fc.ownWeight(w, key);
}
fn ql(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, sub: []const u8) !QLinear {
    const key = try std.fmt.allocPrint(a, "{s}.{s}", .{ pfx, sub });
    defer a.free(key);
    return QLinear.load(w, a, key);
}

fn loadDouble(w: *const Weights, a: std.mem.Allocator, pfx: []const u8) !DoubleBlock {
    return .{
        .norm1_lin = try ql(w, a, pfx, "norm1.linear"),
        .norm1c_lin = try ql(w, a, pfx, "norm1_context.linear"),
        .q = try ql(w, a, pfx, "attn.to_q"),
        .k = try ql(w, a, pfx, "attn.to_k"),
        .v = try ql(w, a, pfx, "attn.to_v"),
        .o = try ql(w, a, pfx, "attn.to_out.0"),
        .add_q = try ql(w, a, pfx, "attn.add_q_proj"),
        .add_k = try ql(w, a, pfx, "attn.add_k_proj"),
        .add_v = try ql(w, a, pfx, "attn.add_v_proj"),
        .add_o = try ql(w, a, pfx, "attn.to_add_out"),
        .nq = try qkNorm(w, a, pfx, "attn.norm_q.weight"),
        .nk = try qkNorm(w, a, pfx, "attn.norm_k.weight"),
        .naq = try qkNorm(w, a, pfx, "attn.norm_added_q.weight"),
        .nak = try qkNorm(w, a, pfx, "attn.norm_added_k.weight"),
        .ff1 = try ql(w, a, pfx, "ff.linear1"),
        .ff2 = try ql(w, a, pfx, "ff.linear2"),
        .ffc1 = try ql(w, a, pfx, "ff_context.linear1"),
        .ffc2 = try ql(w, a, pfx, "ff_context.linear2"),
    };
}
fn loadSingle(w: *const Weights, a: std.mem.Allocator, pfx: []const u8) !SingleBlock {
    return .{
        .norm_lin = try ql(w, a, pfx, "norm.linear"),
        .q = try ql(w, a, pfx, "attn.to_q"),
        .k = try ql(w, a, pfx, "attn.to_k"),
        .v = try ql(w, a, pfx, "attn.to_v"),
        .nq = try qkNorm(w, a, pfx, "attn.norm_q.weight"),
        .nk = try qkNorm(w, a, pfx, "attn.norm_k.weight"),
        .proj_mlp = try ql(w, a, pfx, "proj_mlp"),
        .proj_out = try ql(w, a, pfx, "proj_out"),
    };
}

pub fn loadDit(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !Dit {
    const dir = try fc.fmtKey(allocator, "{s}/transformer", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    var d: Dit = undefined;
    d.allocator = allocator;
    d.s = s;
    const dbl = fc.countIndexed(&w, allocator, "transformer_blocks.{d}.attn.to_q.weight");
    const sgl = fc.countIndexed(&w, allocator, "single_transformer_blocks.{d}.attn.to_q.weight");
    const has_guid = w.get("time_text_embed.guidance_embedder.linear_1.weight") != null;
    // inner = rows of x_embedder; heads fixed at 24 (head_dim 128) for FLUX.1.
    const inner = fc.rowsOf(&w, "x_embedder.weight");
    d.cfg = .{
        .inner = if (inner != 0) inner else 3072,
        .double_layers = if (dbl != 0) dbl else 19,
        .single_layers = if (sgl != 0) sgl else 38,
        .guidance = has_guid,
    };
    d.cfg.heads = d.cfg.inner / d.cfg.head_dim;
    log.info("[flux1] dit: inner={d} heads={d}x{d} double={d} single={d} guidance={}\n", .{ d.cfg.inner, d.cfg.heads, d.cfg.head_dim, d.cfg.double_layers, d.cfg.single_layers, d.cfg.guidance });

    d.x_embedder = try QLinear.load(&w, allocator, "x_embedder");
    d.context_embedder = try QLinear.load(&w, allocator, "context_embedder");
    d.ts_lin1 = try QLinear.load(&w, allocator, "time_text_embed.timestep_embedder.linear_1");
    d.ts_lin2 = try QLinear.load(&w, allocator, "time_text_embed.timestep_embedder.linear_2");
    d.txt_lin1 = try QLinear.load(&w, allocator, "time_text_embed.text_embedder.linear_1");
    d.txt_lin2 = try QLinear.load(&w, allocator, "time_text_embed.text_embedder.linear_2");
    if (has_guid) {
        d.guid_lin1 = try QLinear.load(&w, allocator, "time_text_embed.guidance_embedder.linear_1");
        d.guid_lin2 = try QLinear.load(&w, allocator, "time_text_embed.guidance_embedder.linear_2");
    } else {
        d.guid_lin1 = null;
        d.guid_lin2 = null;
    }
    d.norm_out_lin = try QLinear.load(&w, allocator, "norm_out.linear");
    d.proj_out = try QLinear.load(&w, allocator, "proj_out");
    d.doubles = try allocator.alloc(DoubleBlock, d.cfg.double_layers);
    for (d.doubles, 0..) |*b, i| {
        const pfx = try fc.fmtKey(allocator, "transformer_blocks.{d}", .{i});
        defer allocator.free(pfx);
        b.* = try loadDouble(&w, allocator, pfx);
    }
    d.singles = try allocator.alloc(SingleBlock, d.cfg.single_layers);
    for (d.singles, 0..) |*b, i| {
        const pfx = try fc.fmtKey(allocator, "single_transformer_blocks.{d}", .{i});
        defer allocator.free(pfx);
        b.* = try loadSingle(&w, allocator, pfx);
    }
    return d;
}

// ── runtime LoRA (unfused, STACKED — summed at forward, never merged) ──

/// Attach every matching adapter in `stack` to its FLUX.1 DiT linear. Module
/// keys mirror `lora.flux1_table`'s canonical targets. Returns the number of
/// (module, adapter) attachments. Non-owning: `stack` must outlive the attach.
pub fn attachLora(dit: *Dit, stack: *const lora_mod.Stack) u32 {
    detachLora(dit);
    var matched: u32 = 0;
    var kbuf: [128]u8 = undefined;
    var rbuf: [lora_mod.MAX_LORAS]lora_mod.Ref = undefined;
    for (dit.doubles, 0..) |*b, i| {
        const mods = .{
            .{ "attn.to_q", &b.q },             .{ "attn.to_k", &b.k },
            .{ "attn.to_v", &b.v },             .{ "attn.to_out", &b.o },
            .{ "attn.add_q_proj", &b.add_q },   .{ "attn.add_k_proj", &b.add_k },
            .{ "attn.add_v_proj", &b.add_v },   .{ "attn.to_add_out", &b.add_o },
            .{ "ff.linear1", &b.ff1 },          .{ "ff.linear2", &b.ff2 },
            .{ "ff_context.linear1", &b.ffc1 }, .{ "ff_context.linear2", &b.ffc2 },
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
        const mods = .{
            .{ "attn.to_q", &b.q },       .{ "attn.to_k", &b.k },       .{ "attn.to_v", &b.v },
            .{ "proj_mlp", &b.proj_mlp }, .{ "proj_out", &b.proj_out },
        };
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
        inline for (.{ &b.q, &b.k, &b.v, &b.o, &b.add_q, &b.add_k, &b.add_v, &b.add_o, &b.ff1, &b.ff2, &b.ffc1, &b.ffc2 }) |lin| lin.clearLoraRefs();
    }
    for (dit.singles) |*b| {
        inline for (.{ &b.q, &b.k, &b.v, &b.proj_mlp, &b.proj_out }) |lin| lin.clearLoraRefs();
    }
}

// ── latent packing + ids + schedule ──

/// Pack noise [1,16,H8,W8] → tokens [1, (H8/2)*(W8/2), 64].
/// diffusers _pack_latents: [B,C,h/2,2,w/2,2] → [B,h/2,w/2,C,2,2] → [B,N,C*4].
fn packLatents(noise: mlx.mlx_array, lh: c_int, lw: c_int, s: S) !mlx.mlx_array {
    const r = try reshape(noise, &[_]c_int{ 1, 16, @divExact(lh, 2), 2, @divExact(lw, 2), 2 }, s);
    defer _ = mlx.mlx_array_free(r);
    const t = try transpose(r, &[_]c_int{ 0, 2, 4, 1, 3, 5 }, s);
    defer _ = mlx.mlx_array_free(t);
    var c = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&c, t, false, s));
    defer _ = mlx.mlx_array_free(c);
    return reshape(c, &[_]c_int{ 1, @divExact(lh, 2) * @divExact(lw, 2), 64 }, s);
}
/// Unpack tokens [1,N,64] → latents [1,16,H8,W8]. Inverse of packLatents.
fn unpackLatents(tokens: mlx.mlx_array, lh: c_int, lw: c_int, s: S) !mlx.mlx_array {
    const r = try reshape(tokens, &[_]c_int{ 1, @divExact(lh, 2), @divExact(lw, 2), 16, 2, 2 }, s);
    defer _ = mlx.mlx_array_free(r);
    const t = try transpose(r, &[_]c_int{ 0, 3, 1, 4, 2, 5 }, s);
    defer _ = mlx.mlx_array_free(t);
    var c = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&c, t, false, s));
    defer _ = mlx.mlx_array_free(c);
    return reshape(c, &[_]c_int{ 1, 16, lh, lw }, s);
}
/// img_ids [N,3]: (0, row, col) over the (H8/2)×(W8/2) grid.
fn buildImgIds(a: std.mem.Allocator, gh: u32, gw: u32) ![]i32 {
    const ids = try a.alloc(i32, gh * gw * 3);
    var idx: usize = 0;
    for (0..gh) |r| {
        for (0..gw) |cc| {
            ids[idx + 0] = 0;
            ids[idx + 1] = @intCast(r);
            ids[idx + 2] = @intCast(cc);
            idx += 3;
        }
    }
    return ids;
}
fn buildTxtIds(a: std.mem.Allocator, seq: usize) ![]i32 {
    const ids = try a.alloc(i32, seq * 3);
    @memset(ids, 0);
    return ids;
}

/// FLUX.1 flow-match sigmas. dev applies the dynamic shift (calculate_shift),
/// schnell does not. Returns sigmas[steps+1] (trailing 0).
fn computeSigmas(a: std.mem.Allocator, image_seq_len: u32, steps: u32, shift: bool) ![]f32 {
    const sig = try a.alloc(f32, steps + 1);
    const fsteps: f64 = @floatFromInt(steps);
    // base sigmas = linspace(1.0, 1/steps, steps)
    var mu: f64 = 0;
    if (shift) {
        const base_seq = 256.0;
        const max_seq = 4096.0;
        const base_shift = 0.5;
        const max_shift = 1.15;
        const isl: f64 = @floatFromInt(image_seq_len);
        const m = (max_shift - base_shift) / (max_seq - base_seq);
        const b = base_shift - m * base_seq;
        mu = m * isl + b;
    }
    const emu = std.math.exp(mu);
    for (0..steps) |i| {
        const s0 = 1.0 - @as(f64, @floatFromInt(i)) * (1.0 - 1.0 / fsteps) / (fsteps - 1.0);
        const sv = if (shift) emu / (emu + (1.0 / s0 - 1.0)) else s0;
        sig[i] = @floatCast(sv);
    }
    sig[steps] = 0.0;
    return sig;
}

// ── full pipeline ──

pub const Flux1 = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    s: S,
    model_dir: []u8,
    te: ?t5.T5Encoder,
    clip: ?clip_mod.ClipEncoder,
    tok: t5tok.T5Tokenizer,
    clip_ids: []i32, // CLIP token ids buffer (encoded per prompt)
    dit: Dit,
    vae: vae_mod.Vae,
    guidance_scale: f32,
    is_schnell: bool,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Flux1 {
        const s = mlx.mlx_default_gpu_stream_new();
        var self: Flux1 = undefined;
        self.allocator = allocator;
        self.io = io;
        self.s = s;
        self.model_dir = try allocator.dupe(u8, model_dir);
        errdefer allocator.free(self.model_dir);
        self.tok = try t5tok.T5Tokenizer.load(io, allocator, model_dir);
        errdefer self.tok.deinit();
        self.clip_ids = &.{};
        self.dit = try loadDit(io, allocator, s, model_dir);
        errdefer self.dit.deinit();
        self.vae = try vae_mod.loadVae(io, allocator, s, model_dir);
        errdefer self.vae.deinit();
        // Text encoders are loaded lazily per-generation then freed (staged
        // residency): they are ~2.7 GB (T5) and only needed before denoise.
        self.te = null;
        self.clip = null;
        self.is_schnell = !self.dit.cfg.guidance;
        self.guidance_scale = if (self.is_schnell) 0.0 else 3.5;
        return self;
    }

    pub fn deinit(self: *Flux1) void {
        if (self.te) |*t| t.deinit();
        if (self.clip) |*c| c.deinit();
        self.tok.deinit();
        if (self.clip_ids.len > 0) self.allocator.free(self.clip_ids);
        self.dit.deinit();
        self.vae.deinit();
        self.allocator.free(self.model_dir);
        _ = mlx.mlx_stream_free(self.s);
    }

    /// Encode the prompt with T5 (context) + CLIP (pooled). Materializes both
    /// then frees the encoders (staged residency).
    fn encodePrompt(self: *Flux1, prompt: []const u8) !struct { context: mlx.mlx_array, pooled: mlx.mlx_array } {
        const a = self.allocator;
        // T5
        var t5_enc = try t5tok.T5Tokenizer.encode(&self.tok, prompt, if (self.is_schnell) 256 else T5_MAX_LEN);
        defer t5_enc.deinit();
        if (self.te == null) self.te = try t5.loadEncoder(self.io, a, self.s, self.model_dir);
        const context = try self.te.?.encode(t5_enc.ids);
        errdefer _ = mlx.mlx_array_free(context);
        _ = mlx.mlx_array_eval(context);
        // CLIP — reuse the T5 tokenizer's ids? No: CLIP needs its own tokenizer.
        // The CLIP tower pools at the EOS position; we tokenize with the CLIP
        // tokenizer loaded from tokenizer/. (Loaded lazily via clipEncode.)
        const pooled = try self.clipEncode(prompt);
        errdefer _ = mlx.mlx_array_free(pooled);
        _ = mlx.mlx_array_eval(pooled);
        // Free encoders now (before the DiT denoise) to reclaim ~3 GB.
        if (self.te) |*t| {
            t.deinit();
            self.te = null;
        }
        if (self.clip) |*c| {
            c.deinit();
            self.clip = null;
        }
        return .{ .context = context, .pooled = pooled };
    }

    /// Tokenize with the CLIP tokenizer + run the CLIP tower → pooled [1,1,768].
    fn clipEncode(self: *Flux1, prompt: []const u8) !mlx.mlx_array {
        const a = self.allocator;
        if (self.clip == null) self.clip = try clip_mod.loadEncoder(self.io, a, self.s, self.model_dir);
        const ids = try clipTokenize(self.io, a, self.model_dir, prompt);
        defer a.free(ids);
        return self.clip.?.encodePooled(ids);
    }

    /// latents [1,3,H,W] in [0,1].
    pub fn generateImage(self: *Flux1, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, progress: ?sse.Progress) !mlx.mlx_array {
        _ = allocator;
        const s = self.s;
        const a = self.allocator;
        const lh: u32 = height / 8; // VAE latent H
        const lw: u32 = width / 8;
        const gh: u32 = lh / 2; // packed grid
        const gw: u32 = lw / 2;
        const n_img = gh * gw;

        const cond = try self.encodePrompt(prompt);
        const context = cond.context;
        defer _ = mlx.mlx_array_free(context);
        const pooled = cond.pooled;
        defer _ = mlx.mlx_array_free(pooled);

        // noise [1,16,lh,lw]
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        const nsh = [_]c_int{ 1, 16, @intCast(lh), @intCast(lw) };
        var noise = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(noise);
        try mlx.check(mlx.mlx_random_normal(&noise, &nsh, 4, .float32, 0.0, 1.0, key, s));
        const noise_bf = try astype(noise, .bfloat16, s);
        defer _ = mlx.mlx_array_free(noise_bf);
        var latents = try packLatents(noise_bf, @intCast(lh), @intCast(lw), s);
        errdefer _ = mlx.mlx_array_free(latents);

        const img_ids = try buildImgIds(a, gh, gw);
        defer a.free(img_ids);
        const txt_ids = try buildTxtIds(a, @intCast(mlx.getShape(context)[1]));
        defer a.free(txt_ids);

        const sig = try computeSigmas(a, n_img, steps, !self.is_schnell);
        defer a.free(sig);
        const guid = self.guidance_scale * 1000.0;

        for (0..steps) |t| {
            if (progress) |p| if (p.cancelled()) return error.Cancelled;
            const t_embed = sig[t] * 1000.0;
            const nz = try self.dit.forward(latents, context, pooled, t_embed, guid, img_ids, txt_ids);
            defer _ = mlx.mlx_array_free(nz);
            const dt = sig[t + 1] - sig[t];
            const dta = mlx.mlx_array_new_float(dt);
            defer _ = mlx.mlx_array_free(dta);
            const step = try mulA(nz, dta, s);
            defer _ = mlx.mlx_array_free(step);
            const nl = try addA(latents, step, s);
            _ = mlx.mlx_array_free(latents);
            latents = nl;
            _ = mlx.mlx_array_eval(latents);
            if (progress) |p| p.emit("Generating", @intCast(t + 1), steps);
        }
        if (progress) |p| p.emit("Decoding image", steps, steps);

        // unpack → [1,16,lh,lw] → VAE decode → [1,3,H,W] in [-1,1]
        const unp = try unpackLatents(latents, @intCast(lh), @intCast(lw), s);
        _ = mlx.mlx_array_free(latents);
        latents = .{ .ctx = null };
        defer _ = mlx.mlx_array_free(unp);
        const decoded = try self.vae.decode(unp);
        defer _ = mlx.mlx_array_free(decoded);
        // [-1,1] → [0,1]: clip(x/2 + 0.5, 0, 1)
        const half = mlx.mlx_array_new_float(0.5);
        defer _ = mlx.mlx_array_free(half);
        const df = try astype(decoded, .float32, s);
        defer _ = mlx.mlx_array_free(df);
        const scaled = try mulA(df, half, s);
        defer _ = mlx.mlx_array_free(scaled);
        const shifted = try addA(scaled, half, s);
        defer _ = mlx.mlx_array_free(shifted);
        const lo = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(lo);
        const hi = mlx.mlx_array_new_float(1.0);
        defer _ = mlx.mlx_array_free(hi);
        var clo = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(clo);
        try mlx.check(mlx.mlx_maximum(&clo, shifted, lo, s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_minimum(&out, clo, hi, s));
        return out;
    }
};

/// Tokenize with the CLIP BPE tokenizer (tokenizer/), pad/truncate to 77 with
/// the eos/pad token. Uses the shared byte-level BPE loader in tokenizer.zig.
fn clipTokenize(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, prompt: []const u8) ![]i32 {
    const tok_mod = @import("tokenizer.zig");
    const dir = try std.fmt.allocPrint(a, "{s}/tokenizer", .{model_dir});
    defer a.free(dir);
    var tk = try tok_mod.loadTokenizer(io, a, dir);
    defer tk.deinit();
    const bos: i32 = 49406;
    const eos: i32 = 49407;
    const enc = try tk.encode(a, prompt);
    defer a.free(enc);
    var ids = try a.alloc(i32, CLIP_MAX_LEN);
    @memset(ids, eos); // CLIP pads with eos
    ids[0] = bos;
    var w: usize = 1;
    for (enc) |id| {
        if (w >= CLIP_MAX_LEN - 1) break;
        ids[w] = @intCast(id);
        w += 1;
    }
    ids[w] = eos;
    return ids;
}

// ── tests ──
const testing = std.testing;

test "flux1 sigmas: schnell unshifted linspace, dev shifted+monotonic" {
    const a = testing.allocator;
    // schnell (no shift): linspace(1, 1/4, 4) then 0.
    const sch = try computeSigmas(a, 1024, 4, false);
    defer a.free(sch);
    try testing.expectApproxEqAbs(@as(f32, 1.0), sch[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.25), sch[3], 1e-5);
    try testing.expectEqual(@as(f32, 0.0), sch[4]);
    // dev (shift): still starts at 1, ends 0, strictly decreasing.
    const dev = try computeSigmas(a, 4096, 28, true);
    defer a.free(dev);
    try testing.expectApproxEqAbs(@as(f32, 1.0), dev[0], 1e-5);
    try testing.expectEqual(@as(f32, 0.0), dev[28]);
    for (0..28) |i| try testing.expect(dev[i] > dev[i + 1]);
}

// Full-pipeline smoke — env-gated on a real pack. Loads the whole FLUX.1 stack
// (tokenizer → T5 → CLIP → MMDiT → VAE), generates a small image, and asserts
// the output is finite and in [0,1]. Optionally writes a PPM for eyeballing.
// Numerical parity vs the reference pipeline is a separate fixture; this proves
// the pipeline runs end-to-end on real weights.
//   FLUX1_TEST_MODEL = pack dir   FLUX1_PPM = optional output.ppm
//   FLUX1_STEPS (default 4)  FLUX1_SIZE (default 256)  FLUX1_PROMPT
test "flux1 end-to-end smoke generates a valid image" {
    const model_p = std.c.getenv("FLUX1_TEST_MODEL") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const steps: u32 = if (std.c.getenv("FLUX1_STEPS")) |e| std.fmt.parseInt(u32, std.mem.span(e), 10) catch 4 else 4;
    const size: u32 = if (std.c.getenv("FLUX1_SIZE")) |e| std.fmt.parseInt(u32, std.mem.span(e), 10) catch 256 else 256;
    const prompt = if (std.c.getenv("FLUX1_PROMPT")) |e| std.mem.span(e) else "a photo of a cat";

    var f = try Flux1.load(io, a, std.mem.span(model_p));
    defer f.deinit();
    const img = try f.generateImage(a, prompt, size, size, 42, steps, null);
    defer _ = mlx.mlx_array_free(img);
    _ = mlx.mlx_array_eval(img);
    const n: usize = @intCast(mlx.mlx_array_size(img));
    try testing.expectEqual(@as(usize, 3 * size * size), n);
    const data = mlx.mlx_array_data_float32(img) orelse return error.NoData;
    var lo: f32 = 1e9;
    var hi: f32 = -1e9;
    var sum: f64 = 0;
    for (0..n) |i| {
        const v = data[i];
        try testing.expect(std.math.isFinite(v));
        lo = @min(lo, v);
        hi = @max(hi, v);
        sum += v;
    }
    std.debug.print("[flux1-e2e] n={d} min={d:.3} max={d:.3} mean={d:.3}\n", .{ n, lo, hi, sum / @as(f64, @floatFromInt(n)) });
    try testing.expect(lo >= 0.0 and hi <= 1.0);
    // A real image is not a constant field.
    try testing.expect(hi - lo > 0.05);
    if (std.c.getenv("FLUX1_PPM")) |pp| try @import("flux.zig").writePpm(io, a, img, std.mem.span(pp), f.s);
}

test "flux1 pack/unpack latents is identity on the token grid" {
    // Structural: (H8/2)*(W8/2) tokens of width 64 for a 64×64 latent.
    const gh: u32 = 16;
    const gw: u32 = 16;
    const ids = try buildImgIds(testing.allocator, gh, gw);
    defer testing.allocator.free(ids);
    try testing.expectEqual(gh * gw * 3, ids.len);
    // last token is (0, gh-1, gw-1)
    try testing.expectEqual(@as(i32, 0), ids[(gh * gw - 1) * 3 + 0]);
    try testing.expectEqual(@as(i32, @intCast(gh - 1)), ids[(gh * gw - 1) * 3 + 1]);
    try testing.expectEqual(@as(i32, @intCast(gw - 1)), ids[(gh * gw - 1) * 3 + 2]);
}
