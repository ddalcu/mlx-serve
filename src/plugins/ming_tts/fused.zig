//! Fused Metal kernels for Ming's per-step decode: the DiT stacks (CFM flow head, aggregator) and the Qwen2 LM's
//! one-token step. Sources: `kernels/`.
//!
//! The DiT runs on 6 rows (CFG batch 2 x seq 3) or 2 rows (aggregator), so every linear is a bandwidth-bound GEMV. One
//! DiT block becomes 4 dispatches, each reading its weights once:
//!
//!   A  rmsnorm1 + q/k/v linear + bias + rotary, then per head attention   -> QKV [3, R, 1024], ATT [R, 1024]
//!   B  out linear + residual
//!   C  rmsnorm2 + ff1 linear + bias + gelu_approx                         -> [R, 4096]
//!   D  ff2 linear + bias + residual
//!
//! plus, per Euler step of the CFM, E (x embedder + input row assembly) and F (final norm + linear on the last row +
//! CFG combine + Euler update). The LM step is 5 dispatches per layer: LA norm + q/k/v + rope, LB attention + cache
//! append, LC o_proj + residual, LD norm + gate/up + swiglu, LE down + residual.
//!
//! Storage is bf16, accumulation f32, and every value the op chain rounds to bf16 is rounded here at the same point.
//! What differs: rmsnorm is folded into the GEMV (1/rms applied to the f32 dot product), f32 summation order, a two-pass
//! softmax, gelu's x**3 as x*x*x, and swiglu's sigmoid rounded once. `MLX_SERVE_MING_FUSED=0` runs the op chain.
//!
//! GEMV layout: a threadgroup of SG simdgroups owns NO output columns for all R rows; the simdgroups take 128-wide
//! blocks of K round robin, lanes read 4 consecutive bf16 per block (one 8-byte load per weight row), and partial sums
//! meet in threadgroup memory. R is capped at 6 (attention output in threadgroup memory, and register pressure).

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const kernels = @import("../../mlx/kernels.zig");

const Arr = mlx.mlx_array;
const S = mlx.mlx_stream;

const D = 1024;
const FF = 4096;
const MAX_ROWS = 6;
/// GEMV tile: output columns per threadgroup, simdgroups per threadgroup (they split K).
const Tile = struct { no: c_int, sg: c_int };

var enabled_cache: ?bool = null;

/// Default on: only `MLX_SERVE_MING_FUSED=0` turns the kernels off.
pub fn enabled() bool {
    if (enabled_cache) |v| return v;
    const raw = env.span(.MLX_SERVE_MING_FUSED);
    const v = raw == null or !std.mem.startsWith(u8, raw.?, "0");
    enabled_cache = v;
    return v;
}

fn src(comptime name: []const u8) @TypeOf(kernels.source("plugins/ming_tts/kernels/" ++ name ++ ".metal")) {
    return kernels.source("plugins/ming_tts/kernels/" ++ name ++ ".metal");
}

pub const HEADER = src("header");
const PRELUDE = src("prelude");

/// Partial dot products of the threadgroup's NO output columns over all R rows, summed into `red`. `wrow` is column
/// n's weight row; `xv` loads float4 of row r at i. With `norm`, the rows are rms-normed on the fly: the sum of squares
/// rides along in `ss`, the norm weight NW scales x here, and 1/rms is applied in the epilogue.
fn gemv(comptime k: usize, comptime wrow: []const u8, comptime xv: []const u8, comptime norm: bool) []const u8 {
    return std.fmt.comptimePrint(src("gemv.fmt"), .{
        k,
        wrow,
        xv,
        if (norm) "const float4 nw = LDW(NW, i); for (int r = 0; r < R; r++) { ss[r] += dot(xv[r], xv[r]); xv[r] *= nw; }" else "",
        if (norm) "{ const float v = simd_sum(ss[r]); if (lane == 0) ssr[int(sg) * R + r] = v; }" else "",
    });
}

pub const SRC_QKV = PRELUDE ++ src("dit_qkv_head") ++ gemv(1024, "W + (col0 + n) * 1024", "LDW(X, r * 1024 + i)", true) ++ src("dit_qkv_tail");
pub const SRC_ATTN = PRELUDE ++ gemv(1024, "WO + (ob + n) * 1024", "LDW(ATT, r * 1024 + i)", false) ++ src("dit_attn_out");
pub const SRC_FF1 = PRELUDE ++ gemv(1024, "W1 + (ob + n) * 1024", "LDW(X, r * 1024 + i)", true) ++ src("dit_ff1");
pub const SRC_FF2 = PRELUDE ++ gemv(4096, "W2 + (ob + n) * 4096", "LDW(HH, r * 4096 + i)", false) ++ src("dit_ff2");
pub const SRC_EMBED = src("cfm_embed");
pub const SRC_FINAL = src("cfm_final");
pub const SRC_LM_QKV = PRELUDE ++ src("lm_qkv_head") ++ gemv(896, "W + COL(n) * 896", "LDW(X, i)", true) ++ src("lm_qkv_tail");
pub const SRC_LM_ATTN = src("lm_attn");
pub const SRC_LM_O = PRELUDE ++ gemv(896, "WO + (ob + n) * 896", "LDW(O, i)", false) ++ src("lm_o");
pub const SRC_LM_UP = PRELUDE ++ src("lm_up_head") ++ gemv(896, "(n < NO / 2 ? WG + (c0 + n) * 896 : WU + (c0 + n - NO / 2) * 896)", "LDW(X, i)", true) ++ src("lm_up_tail");
pub const SRC_LM_DOWN = PRELUDE ++ gemv(4864, "WD + (ob + n) * 4864", "LDW(HH, i)", false) ++ src("lm_down");

const Kernel = struct {
    name: [*:0]const u8,
    ins: []const [*:0]const u8,
    outs: []const [*:0]const u8 = &.{"OUT"},
    src: [*:0]const u8,
    tile: Tile = .{ .no = 8, .sg = 4 },
    k: ?mlx.mlx_fast_metal_kernel = null,

    fn get(self: *Kernel) !mlx.mlx_fast_metal_kernel {
        if (self.k) |k| return k;
        const iv = mlx.mlx_vector_string_new_data(self.ins.ptr, self.ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(self.outs.ptr, self.outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new(self.name, iv, ov, self.src, HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        self.k = k;
        return k;
    }
};

var k_qkv: Kernel = .{ .name = "ming_dit_qkv", .ins = &.{ "X", "NW", "WQ", "WK", "WV", "BQ", "BK", "BV", "COS", "SIN", "CNT" }, .outs = &.{ "OUT", "ATT" }, .src = SRC_QKV, .tile = .{ .no = 2, .sg = 4 } };
var k_attn: Kernel = .{ .name = "ming_dit_attn_out", .ins = &.{ "ATT", "X", "WO", "BO" }, .src = SRC_ATTN, .tile = .{ .no = 2, .sg = 4 } };
var k_ff1: Kernel = .{ .name = "ming_dit_ff1", .ins = &.{ "X", "NW", "W1", "B1" }, .src = SRC_FF1, .tile = .{ .no = 2, .sg = 4 } };
var k_ff2: Kernel = .{ .name = "ming_dit_ff2", .ins = &.{ "HH", "X", "W2", "B2" }, .src = SRC_FF2, .tile = .{ .no = 2, .sg = 4 } };
var k_embed: Kernel = .{ .name = "ming_cfm_embed", .ins = &.{ "Y", "WX", "BX", "TE", "CE", "XH" }, .src = SRC_EMBED };
var k_lm_qkv: Kernel = .{ .name = "ming_lm_qkv", .ins = &.{ "X", "NW", "WQ", "WK", "WV", "BQ", "BK", "BV", "POS" }, .src = SRC_LM_QKV, .tile = .{ .no = 2, .sg = 4 } };
var k_lm_attn: Kernel = .{ .name = "ming_lm_attn", .ins = &.{ "QKV", "KC", "VC" }, .outs = &.{ "OUT", "KN", "VN" }, .src = SRC_LM_ATTN };
var k_lm_o: Kernel = .{ .name = "ming_lm_o", .ins = &.{ "O", "X", "WO" }, .src = SRC_LM_O, .tile = .{ .no = 2, .sg = 4 } };
var k_lm_up: Kernel = .{ .name = "ming_lm_up", .ins = &.{ "X", "NW", "WG", "WU" }, .src = SRC_LM_UP, .tile = .{ .no = 4, .sg = 4 } };
var k_lm_down: Kernel = .{ .name = "ming_lm_down", .ins = &.{ "HH", "X", "WD" }, .src = SRC_LM_DOWN, .tile = .{ .no = 2, .sg = 4 } };
var k_final: Kernel = .{ .name = "ming_cfm_final", .ins = &.{ "X", "NW", "WF", "BF", "Y", "DTT", "SQ", "SDE" }, .src = SRC_FINAL };

const Tmpl = struct { [*:0]const u8, c_int };

/// Run one kernel → its bf16 outputs (owned by the caller) in `outs`. With
/// `n_cols`, a GEMV kernel: n_cols / tile.no threadgroups of tile.sg
/// simdgroups; otherwise `tgs` threadgroups of `tg` threads.
fn runN(kr: *Kernel, inputs: []const Arr, shapes: []const []const c_int, n_cols: ?c_int, tgs: c_int, tg_threads: c_int, tmpl: []const Tmpl, s: S, outs: []Arr) !void {
    const k = try kr.get();
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (shapes) |sh| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, sh.ptr, sh.len, .bfloat16));
    var n_tg = tgs;
    var tg = tg_threads;
    if (n_cols) |n| {
        const tl = kr.tile;
        n_tg = @divExact(n, tl.no);
        tg = 32 * tl.sg;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NO", tl.no));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "SG", tl.sg));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, n_tg * tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, tg, 1, 1));
    for (tmpl) |tp| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, tp[0], tp[1]));
    const v = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, k, v, c, s));
    for (outs, 0..) |*out, i| {
        out.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(out, o, i));
    }
}

fn run(kr: *Kernel, inputs: []const Arr, out_shape: []const c_int, n_cols: ?c_int, tgs: c_int, tmpl: []const Tmpl, s: S) !Arr {
    var out: [1]Arr = undefined;
    try runN(kr, inputs, &.{out_shape}, n_cols, tgs, 128, tmpl, s, &out);
    return out[0];
}

/// Weights of one DiT block, original [out, in] layout (contiguous).
pub const Block = struct {
    norm1: Arr,
    norm2: Arr,
    wq: Arr,
    wk: Arr,
    wv: Arr,
    bq: Arr,
    bk: Arr,
    bv: Arr,
    wo: Arr,
    bo: Arr,
    w1: Arr,
    b1: Arr,
    w2: Arr,
    b2: Arr,
    /// int32 [1024] (big enough to be bound as device memory) counter for A's last-threadgroup election, shared by
    /// the blocks of one stack (zero at load; only ever incremented by A).
    cnt: Arr,
};

/// Whether x [B, L, 1024] fits the fused DiT kernels.
pub fn eligible(x: Arr, s: S) bool {
    if (!enabled() or !mlx.streamIsGpu(s)) return false;
    const sh = mlx.getShape(x);
    return sh.len == 3 and sh[2] == D and sh[0] * sh[1] <= MAX_ROWS and mlx.mlx_array_dtype(x) == .bfloat16;
}

pub const BlockOut = struct { qkv: Arr, att: Arr, x1: Arr, hh: Arr, x2: Arr };

fn qkvAtt(b: *const Block, x: Arr, cos: Arr, sin: Arr, t: []const Tmpl, s: S) ![2]Arr {
    const sh = mlx.getShape(x);
    const R = sh[0] * sh[1];
    const tl = k_qkv.tile;
    const ta: []const Tmpl = &.{ t[0], t[1], .{ "NTG", @divExact(3 * 64, tl.no) } };
    var o: [2]Arr = undefined;
    try runN(&k_qkv, &.{ x, b.norm1, b.wq, b.wk, b.wv, b.bq, b.bk, b.bv, cos, sin, b.cnt }, &.{ &.{ 3, R, D }, &.{ R, D } }, 3 * D, 0, 128, ta, s, &o);
    return o;
}

/// One DiT block, 4 dispatches. Intermediates are returned so the caller's
/// graph scope frees them; `x2` is the block output [B, L, 1024].
pub fn block(b: *const Block, x: Arr, cos: Arr, sin: Arr, s: S) !BlockOut {
    const sh = mlx.getShape(x);
    const L = sh[1];
    const R = sh[0] * L;
    const t: []const Tmpl = &.{ .{ "R", R }, .{ "L", L } };
    const qa = try qkvAtt(b, x, cos, sin, t, s);
    errdefer for (qa) |y| {
        _ = mlx.mlx_array_free(y);
    };
    const x1 = try run(&k_attn, &.{ qa[1], x, b.wo, b.bo }, sh, D, 0, t, s);
    errdefer _ = mlx.mlx_array_free(x1);
    const hh = try run(&k_ff1, &.{ x1, b.norm2, b.w1, b.b1 }, &.{ R, FF }, FF, 0, t, s);
    errdefer _ = mlx.mlx_array_free(hh);
    const x2 = try run(&k_ff2, &.{ hh, x1, b.w2, b.b2 }, sh, D, 0, t, s);
    return .{ .qkv = qa[0], .att = qa[1], .x1 = x1, .hh = hh, .x2 = x2 };
}

/// CFM input rows h [2, 3, 1024] from latent y [1, 1, 64].
pub fn embed(y: Arr, wx: Arr, bx: Arr, te: Arr, ce: Arr, xh: Arr, s: S) !Arr {
    return run(&k_embed, &.{ y, wx, bx, te, ce, xh }, &.{ 2, 3, D }, null, D / 128, &.{}, s);
}

/// Euler step `step` from DiT output h [2, L, 1024] → new latent [1, 1, 64].
/// dtt / sq: [FLOW_STEPS] step sizes and sqrt|dt|; sde: [FLOW_STEPS, 1, 1, 64].
pub fn final(h: Arr, nw: Arr, wf: Arr, bf: Arr, y: Arr, dtt: Arr, sq: Arr, sde: Arr, step: usize, cfg: f32, shift: f32, s: S) !Arr {
    const L = mlx.getShape(h)[1];
    return run(&k_final, &.{ h, nw, wf, bf, y, dtt, sq, sde }, &.{ 1, 1, 64 }, null, 64 / 4, &.{
        .{ "L", L },
        .{ "S", @intCast(step) },
        .{ "CFG", @bitCast(cfg) },
        .{ "SHIFT", @bitCast(shift) },
    }, s);
}

/// Weights of one Qwen2 layer (q/k/v biased; o, gate, up, down not).
pub const LmLayer = struct {
    ln1: Arr,
    ln2: Arr,
    wq: Arr,
    wk: Arr,
    wv: Arr,
    bq: Arr,
    bk: Arr,
    bv: Arr,
    wo: Arr,
    wg: Arr,
    wu: Arr,
    wd: Arr,
};

pub const LmOut = struct { qkv: Arr, o: Arr, x1: Arr, hh: Arr, x2: Arr, k: Arr, v: Arr };

/// One Qwen2 decode layer on x [1, 1, 896] at position `pos` (the cache
/// length) with cache kc/vc [1, 2, pos, 64]. Returns the layer output `x2`
/// and the appended cache `k`/`v`; the rest are intermediates for the
/// caller's scope. `log2_theta`: log2 of the rope base, as MLX passes it.
pub fn lmLayer(l: *const LmLayer, x: Arr, pos: Arr, kc: Arr, vc: Arr, log2_theta: f32, s: S) !LmOut {
    var r: LmOut = undefined;
    const sh = mlx.getShape(x);
    const n: c_int = mlx.getShape(kc)[2] + 1;
    if (n > 512) return error.CacheTooLong;
    const t1: []const Tmpl = &.{ .{ "R", 1 }, .{ "LOG2BASE", @bitCast(log2_theta) } };
    const t: []const Tmpl = &.{.{ "R", 1 }};
    r.qkv = try run(&k_lm_qkv, &.{ x, l.ln1, l.wq, l.wk, l.wv, l.bq, l.bk, l.bv, pos }, &.{1152}, 1152, 0, t1, s);
    var three: [3]Arr = undefined;
    try runN(&k_lm_attn, &.{ r.qkv, kc, vc }, &.{ sh, &.{ 1, 2, n, 64 }, &.{ 1, 2, n, 64 } }, null, 14, 256, &.{}, s, &three);
    r.o = three[0];
    r.k = three[1];
    r.v = three[2];
    r.x1 = try run(&k_lm_o, &.{ r.o, x, l.wo }, sh, 896, 0, t, s);
    r.hh = try run(&k_lm_up, &.{ r.x1, l.ln2, l.wg, l.wu }, &.{4864}, 2 * 4864, 0, t, s);
    r.x2 = try run(&k_lm_down, &.{ r.hh, r.x1, l.wd }, sh, 896, 0, t, s);
    return r;
}

// ── tests ──

test "ming_tts fused: on unless the switch says 0" {
    try std.testing.expect(env.span(.MLX_SERVE_MING_FUSED) != null or enabled());
}
