//! SeedVR2 NaDiT — the native-resolution MMDiT that does the restoration.
//!
//! Transcribed from ByteDance-Seed/SeedVR `models/dit_v2/*` (Apache-2.0). Full
//! spec + ranked trap list in `docs/seedvr2-arch.md` §1–2. The window partition
//! is `seedvr2_window.zig` (oracle-pinned separately) and the tensor names are
//! `seedvr2_manifest.zig` (reconciled 635/635 against the real checkpoint).
//!
//! SHAPE MODEL: one batch item, tokens flattened. `vid` is `[Lv, dim]` over a
//! `(T, H, W)` token grid in row-major `t, h, w` order; `txt` is `[Lt, dim]`.
//!
//! FOUR THINGS THE SOURCE DOES NOT SAY OUT LOUD:
//!
//!   1. `vid_out_ada` reuses the ATTN modulation slice. Its own 1-layer view is
//!      malformed and never evaluated — `AdaSingle`'s memo key collides with the
//!      blocks' attn entry and the cache hits. See §1.3.1; this is not a bug to
//!      route around, it is what the weights were trained against.
//!   2. Text tokens are BROADCAST into every window and MEAN-POOLED on the way
//!      out. Attending them in one window, or keeping the last copy, is
//!      plausible and wrong.
//!   3. RoPE is WINDOW-LOCAL: every window restarts at `(txt_len, 0, 0)`. The
//!      video temporal axis is offset by the text length; text sits at
//!      `0..txt_len-1`.
//!   4. Weights live under `.vid`/`.txt` for layers < 10 and under `.all` after.
//!      Every shape matches either way.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const win_mod = @import("seedvr2_window.zig");
const manifest = @import("seedvr2_manifest.zig");
const log = @import("log.zig");

const S = mlx.mlx_stream;
const Weights = model_mod.Weights;

pub const Config = manifest.Config;

/// `num_windows` for every layer in the 3B config.
const NUM_WINDOWS: [3]u32 = .{ 4, 3, 3 };

// ════════════════════════════════════════════════════════════════════════
// Compute dtype
// ════════════════════════════════════════════════════════════════════════

/// The dtype every weight is widened to at load, and every matmul runs in.
///
/// **bf16, because that is the precision the checkpoint IS.** SeedVR2 ships
/// `seedvr2_ema_3b_fp16.safetensors` and the reference casts q/k/v to bf16
/// before flash-attn and takes the bf16 result back (§1.1.2), so an f32 port
/// is not a more faithful one — it is a 2x more expensive one that lands the
/// same distance from the reference and doubles what the 3B costs to hold
/// (6.8 GB of weights become 13.5 GB, which is the difference between loading
/// and not on a 32 GB Mac).
///
/// f32 stays reachable and is what the parity fixtures pin, because a parity
/// test against an f32-dumped oracle must not be reading its own rounding back
/// as agreement. Kill switch for A/Bs and for that test: MLX_SERVE_SEEDVR2_F32=1.
///
/// The VAE is deliberately NOT covered by this: its GroupNorm statistics are
/// per-frame over few elements and it is 1 GB either way, so it stays f32 —
/// the MageFlow rule that only the sensitive component needs the wide dtype,
/// pointed the other way.
pub fn computeDtype() mlx.mlx_dtype {
    return if (forceF32()) mlx.mlx_dtype.float32 else mlx.mlx_dtype.bfloat16;
}

fn forceF32() bool {
    const v = std.c.getenv("MLX_SERVE_SEEDVR2_F32") orelse return false;
    const span = std.mem.span(v);
    return span.len > 0 and span[0] == '1';
}

/// How many bytes the load holds per byte of the SHIPPED fp16 checkpoint.
///
/// The residency estimator needs this: it bills the directory's file sizes,
/// and a load that widens every weight makes that number a lie by exactly this
/// factor (`gen.seedvr2PeakBytes`). 1 under bf16, 2 under f32.
pub fn dtypeWidthRatio() u64 {
    return if (computeDtype() == mlx.mlx_dtype.float32) 2 else 1;
}

// ════════════════════════════════════════════════════════════════════════
// Array helpers
// ════════════════════════════════════════════════════════════════════════

fn contig(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}
fn transpose(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}
fn reshape(x: mlx.mlx_array, shp: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shp.ptr, shp.len, s));
    return o;
}
fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}
fn addA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, a, b, s));
    return o;
}
fn mulA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, a, b, s));
    return o;
}
fn concatA(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}
fn sliceAxis(x: mlx.mlx_array, axis: usize, lo: c_int, hi: c_int, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(x);
    var start: [8]c_int = undefined;
    var stop: [8]c_int = undefined;
    var step: [8]c_int = undefined;
    const nd = shp.len;
    for (0..nd) |i| {
        start[i] = 0;
        stop[i] = @intCast(shp[i]);
        step[i] = 1;
    }
    start[axis] = lo;
    stop[axis] = hi;
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_slice(&o, x, &start, nd, &stop, nd, &step, nd, s));
    return contig(o, s);
}
fn siluA(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&o, x, s));
    defer _ = mlx.mlx_array_free(o);
    return mulA(x, o, s);
}
/// `gelu_approx` — MLX's tanh GELU, which is what the reference's plain MLP
/// calls: `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))`. NOT the erf gelu and
/// NOT `gelu_fast_approx`'s sigmoid form; the three disagree in the third
/// decimal, which over 36 blocks is a visibly different picture.
fn geluTanh(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const c_cube = mlx.mlx_array_new_float(0.044715);
    defer _ = mlx.mlx_array_free(c_cube);
    const c_sqrt = mlx.mlx_array_new_float(0.7978845608028654); // sqrt(2/pi)
    defer _ = mlx.mlx_array_free(c_sqrt);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);

    const x2 = try mulA(x, x, s);
    defer _ = mlx.mlx_array_free(x2);
    const x3 = try mulA(x2, x, s);
    defer _ = mlx.mlx_array_free(x3);
    const cx3 = try mulA(x3, c_cube, s);
    defer _ = mlx.mlx_array_free(cx3);
    const inner = try addA(x, cx3, s);
    defer _ = mlx.mlx_array_free(inner);
    const scaled = try mulA(inner, c_sqrt, s);
    defer _ = mlx.mlx_array_free(scaled);
    var th = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(th);
    try mlx.check(mlx.mlx_tanh(&th, scaled, s));
    const onep = try addA(th, one, s);
    defer _ = mlx.mlx_array_free(onep);
    const hx = try mulA(x, half, s);
    defer _ = mlx.mlx_array_free(hx);
    return mulA(hx, onep, s);
}

fn takeRows(x: mlx.mlx_array, idx: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_take_axis(&o, x, idx, 0, s));
    return o;
}

/// A new owning handle on an existing array, so a branch that passes its input
/// straight through still returns something the caller can free unconditionally.
fn ownArray(x: mlx.mlx_array) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, x));
    return o;
}

fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const a = w.get(key) orelse {
        log.err("[seedvr2-dit] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingSeedVr2DitWeight;
    };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, a));
    return o;
}
fn loadVec(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const name = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(name);
    const raw = try ownWeight(w, name);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, dt, s);
}
/// A `"<prefix>.<suffix>"` vector — the PREFIX-taking twin of `loadVec`, for
/// call sites that already built the prefix once (a `DitLinear`'s bias, a
/// scheme-dependent attn/norm key).
fn loadVecAt(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, comptime suffix: []const u8, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const name = try std.fmt.allocPrint(a, "{s}." ++ suffix, .{prefix});
    defer a.free(name);
    const raw = try ownWeight(w, name);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, dt, s);
}
/// Linear stored `[out, in]` -> `[in, out]` so the hot path is a plain matmul.
fn loadLinT(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const name = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(name);
    const raw = try ownWeight(w, name);
    defer _ = mlx.mlx_array_free(raw);
    const f = try astype(raw, dt, s);
    defer _ = mlx.mlx_array_free(f);
    const t = try transpose(f, &[_]c_int{ 1, 0 }, s);
    defer _ = mlx.mlx_array_free(t);
    return contig(t, s);
}

/// A NaDiT Linear, dense (this project's own bf16/fp16 converter) or
/// affine-quantized (the mlx-community 8-bit mirror). The checkpoint carries
/// no format flag, so it is decided PER TENSOR by the presence of a
/// `.scales` sibling — same primitive as `krea.zig`'s / `mage_flow.zig`'s
/// `MixedLinear`, minus the LoRA arm SeedVR2 has no adapter path for.
/// `(bits, group_size)` are solved from the packed geometry, never assumed,
/// so a mirror at a different width works by construction.
///
/// Kept QUANTIZED at rest and run through `mlx_quantized_matmul` rather than
/// dequantized to the compute dtype at load: the 8-bit pack halves both the
/// download and the resident weight bytes this way, and — unlike a decode
/// loop — the one-step recipe runs the DiT forward once per restore, so
/// there is no per-token bandwidth case for pre-widening to bf16.
const DitLinear = struct {
    quantized: bool,
    /// Dense: pre-transposed `[in, out]` in the compute dtype.
    /// Quantized: packed `[out, in*bits/32]` U32 — `mlx_quantized_matmul`
    /// takes `transpose_w: true` instead, so this stays as shipped.
    w: mlx.mlx_array,
    scales: mlx.mlx_array = .{ .ctx = null },
    biases: mlx.mlx_array = .{ .ctx = null },
    bits: u32 = 0,
    group_size: u32 = 0,

    fn deinit(self: *DitLinear) void {
        _ = mlx.mlx_array_free(self.w);
        if (self.quantized) {
            _ = mlx.mlx_array_free(self.scales);
            _ = mlx.mlx_array_free(self.biases);
        }
    }
};

/// Load a Linear at `<prefix>.weight` (+ `.scales`/`.biases` when quantized),
/// dense or quantized, decided by whether the `.scales` sibling is present.
/// `in_features` is read only on the quantized path, to solve `(bits,
/// group_size)` from the packed shape — the checkpoint states neither.
fn loadMixedLinT(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, in_features: u32, dt: mlx.mlx_dtype, s: S) !DitLinear {
    const wk = try std.fmt.allocPrint(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const sk = try std.fmt.allocPrint(a, "{s}.scales", .{prefix});
    defer a.free(sk);

    if (w.get(sk) != null) {
        const bk = try std.fmt.allocPrint(a, "{s}.biases", .{prefix});
        defer a.free(bk);
        const weight = try ownWeight(w, wk);
        errdefer _ = mlx.mlx_array_free(weight);
        const scales = try ownWeight(w, sk);
        errdefer _ = mlx.mlx_array_free(scales);
        const biases = try ownWeight(w, bk);
        errdefer _ = mlx.mlx_array_free(biases);
        const w_cols: u32 = @intCast(mlx.getShape(weight)[1]); // in*bits/32
        const s_cols: u32 = @intCast(mlx.getShape(scales)[1]); // in/group_size
        const bits: u32 = @intCast(@divExact(32 * w_cols, in_features));
        const gs: u32 = @intCast(@divExact(in_features, s_cols));
        return .{ .quantized = true, .w = weight, .scales = scales, .biases = biases, .bits = bits, .group_size = gs };
    }

    const raw = try ownWeight(w, wk);
    defer _ = mlx.mlx_array_free(raw);
    const f = try astype(raw, dt, s);
    defer _ = mlx.mlx_array_free(f);
    const t = try transpose(f, &[_]c_int{ 1, 0 }, s);
    defer _ = mlx.mlx_array_free(t);
    return .{ .quantized = false, .w = try contig(t, s) };
}

fn linT(x: mlx.mlx_array, wt: *const DitLinear, bias: ?mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    if (wt.quantized) {
        try mlx.check(mlx.mlx_quantized_matmul(&o, x, wt.w, wt.scales, wt.biases, true, mlx.mlx_optional_int.some(@intCast(wt.group_size)), mlx.mlx_optional_int.some(@intCast(wt.bits)), "affine", s));
    } else {
        try mlx.check(mlx.mlx_matmul(&o, x, wt.w, s));
    }
    if (bias) |b| {
        defer _ = mlx.mlx_array_free(o);
        return addA(o, b, s);
    }
    return o;
}

/// RMSNorm over the LAST axis. `weight == null` is the affine-free form the
/// block norms use — all their per-channel affine lives in `AdaSingle`.
///
/// THE STATISTIC IS ALWAYS COMPUTED IN f32, whatever `x`'s dtype. A mean of
/// squares over 2560 channels carries ~3 significant digits in bf16, and every
/// activation in the block is divided by it — this is the one reduction in the
/// forward where the compute dtype is not good enough, and it costs a single
/// elementwise pass to get right. The result is returned in `x`'s own dtype so
/// the caller's arithmetic does not silently promote.
fn rmsNorm(x: mlx.mlx_array, weight: ?mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(x);
    const wide = dt != mlx.mlx_dtype.float32;
    const xf = if (wide) try astype(x, mlx.mlx_dtype.float32, s) else try contig(x, s);
    defer _ = mlx.mlx_array_free(xf);

    const sq = try mulA(xf, xf, s);
    defer _ = mlx.mlx_array_free(sq);
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    const axes = [_]c_int{-1};
    try mlx.check(mlx.mlx_mean_axes(&mean, sq, &axes, axes.len, true, s));
    const e = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(e);
    const me = try addA(mean, e, s);
    defer _ = mlx.mlx_array_free(me);
    var rs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rs);
    try mlx.check(mlx.mlx_rsqrt(&rs, me, s));
    const nf = try mulA(xf, rs, s);
    defer _ = mlx.mlx_array_free(nf);
    const n = if (wide) try astype(nf, dt, s) else try contig(nf, s);
    if (weight) |wt| {
        defer _ = mlx.mlx_array_free(n);
        return mulA(n, wt, s);
    }
    return n;
}

// ════════════════════════════════════════════════════════════════════════
// AdaSingle
// ════════════════════════════════════════════════════════════════════════

/// The timestep embedding split into modulation slots.
///
/// `emb` is `[1, 6*dim]` packed as `(d, l, g)` with **d SLOWEST** — element
/// `(d, l, g)` sits at flat index `d*6 + l*3 + g`, where `l` selects
/// attn(0)/mlp(1) and `g` selects shift(0)/scale(1)/gate(2). A `[3, 2, dim]`
/// view is the natural guess, is the same size, and is transposed.
pub const Modulation = struct {
    /// `[2, 3, dim]` — [layer][shift|scale|gate][channel].
    slots: mlx.mlx_array,
    /// The six `[dim]` slices, MATERIALISED ONCE.
    ///
    /// There is one timestep for the whole forward, so these six vectors are
    /// the same on every one of the 32 blocks. Slicing them per use cost four
    /// slice+contiguous round trips per block per stream — ~250 throwaway
    /// dispatches on the 3B for six values that never change.
    each: [6]mlx.mlx_array,

    fn init(emb: mlx.mlx_array, dim: c_int, s: S) !Modulation {
        // [1, 6*dim] -> [dim, 2, 3] -> [2, 3, dim]
        const r = try reshape(emb, &[_]c_int{ dim, 2, 3 }, s);
        defer _ = mlx.mlx_array_free(r);
        const t = try transpose(r, &[_]c_int{ 1, 2, 0 }, s);
        defer _ = mlx.mlx_array_free(t);
        var m: Modulation = .{ .slots = try contig(t, s), .each = undefined };
        errdefer _ = mlx.mlx_array_free(m.slots);
        var built: usize = 0;
        errdefer for (m.each[0..built]) |p| {
            _ = mlx.mlx_array_free(p);
        };
        for (0..2) |l| {
            for (0..3) |k| {
                m.each[l * 3 + k] = try sliceOf(m.slots, @intCast(l), @intCast(k), s);
                built += 1;
            }
        }
        return m;
    }

    fn sliceOf(slots: mlx.mlx_array, layer: u32, kind: u32, s: S) !mlx.mlx_array {
        const l = try sliceAxis(slots, 0, @intCast(layer), @intCast(layer + 1), s);
        defer _ = mlx.mlx_array_free(l);
        const k = try sliceAxis(l, 1, @intCast(kind), @intCast(kind + 1), s);
        defer _ = mlx.mlx_array_free(k);
        const shp = mlx.getShape(k);
        return reshape(k, &[_]c_int{shp[2]}, s);
    }

    fn deinit(m: *Modulation) void {
        _ = mlx.mlx_array_free(m.slots);
        for (&m.each) |*p| _ = mlx.mlx_array_free(p.*);
    }

    /// `[dim]` for one (layer, kind) — a BORROWED handle into `each`, valid
    /// for the Modulation's lifetime. Callers must not free it.
    fn get(m: *const Modulation, layer: u32, kind: u32) mlx.mlx_array {
        return m.each[layer * 3 + kind];
    }
};

const LAYER_ATTN: u32 = 0;
const LAYER_MLP: u32 = 1;
const KIND_SHIFT: u32 = 0;
const KIND_SCALE: u32 = 1;
const KIND_GATE: u32 = 2;

/// `hid * (scaleA + scaleB) + (shiftA + shiftB)`.
fn adaIn(hid: mlx.mlx_array, m: *const Modulation, layer: u32, shift_b: mlx.mlx_array, scale_b: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sc_a = m.get(layer, KIND_SCALE);
    const sh_a = m.get(layer, KIND_SHIFT);
    const sc = try addA(sc_a, scale_b, s);
    defer _ = mlx.mlx_array_free(sc);
    const sh = try addA(sh_a, shift_b, s);
    defer _ = mlx.mlx_array_free(sh);
    const m1 = try mulA(hid, sc, s);
    defer _ = mlx.mlx_array_free(m1);
    return addA(m1, sh, s);
}

/// `hid * (gateA + gateB)`.
fn adaOut(hid: mlx.mlx_array, m: *const Modulation, layer: u32, gate_b: mlx.mlx_array, s: S) !mlx.mlx_array {
    const g_a = m.get(layer, KIND_GATE);
    const g = try addA(g_a, gate_b, s);
    defer _ = mlx.mlx_array_free(g);
    return mulA(hid, g, s);
}

// ════════════════════════════════════════════════════════════════════════
// mmrope3d
// ════════════════════════════════════════════════════════════════════════

/// Interleaved-pair RoPE tables for one window.
///
/// `freqs` is the checkpoint's stored `[n]` inverse-frequency buffer. Each axis
/// contributes `2n` values (every frequency repeated twice, because the pair
/// convention is `(2i, 2i+1)`), so three axes cover `6n` of `head_dim`. For the
/// 3B that is 126 of 128 — **dims 126:128 are never rotated**, which is a
/// property of the checkpoint, not an oversight to correct.
const Rope = struct {
    cos: mlx.mlx_array,
    sin: mlx.mlx_array,
    rot: c_int,

    fn deinit(r: *Rope) void {
        _ = mlx.mlx_array_free(r.cos);
        _ = mlx.mlx_array_free(r.sin);
    }
};

/// `[len]` positional angles for one axis: `outer(pos, freqs)` with each column
/// duplicated, giving `[len, 2n]`.
fn axisAngles(freqs: mlx.mlx_array, start: c_int, len: c_int, mode: manifest.RopeFreqs, s: S) !mlx.mlx_array {
    var pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos);
    switch (mode) {
        // `lang`: integer positions, offset into the shared txt+vid space.
        .lang => try mlx.check(mlx.mlx_arange(&pos, @floatFromInt(start), @floatFromInt(start + len), 1.0, mlx.mlx_dtype.float32, s)),
        // `pixel`: each axis NORMALISED onto [-1, 1] across its own extent, so
        // there is no `start` to offset by — the reference builds the grid from
        // the exact per-window extent for this reason ("slicing a larger
        // precomputed grid changes spacing and introduces spatial bias"). A
        // single-element axis is the endpoint, matching mx.linspace.
        .pixel => try mlx.check(mlx.mlx_linspace(&pos, -1.0, 1.0, len, mlx.mlx_dtype.float32, s)),
    }
    const p2 = try reshape(pos, &[_]c_int{ len, 1 }, s);
    defer _ = mlx.mlx_array_free(p2);
    const n = mlx.getShape(freqs)[0];
    const f2 = try reshape(freqs, &[_]c_int{ 1, n }, s);
    defer _ = mlx.mlx_array_free(f2);
    const ang = try mulA(p2, f2, s); // [len, n]
    defer _ = mlx.mlx_array_free(ang);
    // repeat each frequency twice along the last axis: [len, n] -> [len, n, 2]
    // -> [len, 2n]. The pair layout is (2i, 2i+1) sharing frequency i.
    const e = try reshape(ang, &[_]c_int{ len, n, 1 }, s);
    defer _ = mlx.mlx_array_free(e);
    const two = try concatA(&[_]mlx.mlx_array{ e, e }, 2, s);
    defer _ = mlx.mlx_array_free(two);
    return reshape(two, &[_]c_int{ len, 2 * n }, s);
}

/// Build the `[T*H*W, 6n]` video angle table for a window of extent
/// `(t, h, w)`. The temporal axis starts at `txt_len` — video positions follow
/// the text in one shared coordinate space — while H and W start at 0. Window
/// extents are LOCAL, so every window restarts from the same origin.
fn videoAngles(freqs: mlx.mlx_array, t: c_int, h: c_int, w: c_int, txt_len: c_int, mode: manifest.RopeFreqs, a: std.mem.Allocator, s: S) !mlx.mlx_array {
    _ = a;
    const n = mlx.getShape(freqs)[0];
    // `txt_len` is 0 whenever the text is not rotated. The offset belongs to
    // the MULTIMODAL rope path, where video positions continue the text's own
    // coordinate space; the video-only path (`rope_on_text` off) starts every
    // axis at the origin. Keying this on the frequency basis instead would
    // give the right answer for both shipped configs and the wrong one for a
    // `lang` checkpoint that does not rotate its text.
    const at = try axisAngles(freqs, txt_len, t, mode, s);
    defer _ = mlx.mlx_array_free(at);
    const ah = try axisAngles(freqs, 0, h, mode, s);
    defer _ = mlx.mlx_array_free(ah);
    const aw = try axisAngles(freqs, 0, w, mode, s);
    defer _ = mlx.mlx_array_free(aw);

    // Broadcast each axis over the other two, then concat on the feature axis.
    const t5 = try reshape(at, &[_]c_int{ t, 1, 1, 2 * n }, s);
    defer _ = mlx.mlx_array_free(t5);
    const h5 = try reshape(ah, &[_]c_int{ 1, h, 1, 2 * n }, s);
    defer _ = mlx.mlx_array_free(h5);
    const w5 = try reshape(aw, &[_]c_int{ 1, 1, w, 2 * n }, s);
    defer _ = mlx.mlx_array_free(w5);

    const shp = [_]c_int{ t, h, w, 2 * n };
    var tb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tb);
    var hb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hb);
    var wb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wb);
    try mlx.check(mlx.mlx_broadcast_to(&tb, t5, &shp, shp.len, s));
    try mlx.check(mlx.mlx_broadcast_to(&hb, h5, &shp, shp.len, s));
    try mlx.check(mlx.mlx_broadcast_to(&wb, w5, &shp, shp.len, s));
    const cat = try concatA(&[_]mlx.mlx_array{ tb, hb, wb }, 3, s);
    defer _ = mlx.mlx_array_free(cat);
    return reshape(cat, &[_]c_int{ t * h * w, 6 * n }, s);
}

/// `[txt_len, 6n]` — the 1-D language table TILED THREE TIMES, not a 3-axis
/// table. Text has one position; it simply has to fill the same width.
fn textAngles(freqs: mlx.mlx_array, txt_len: c_int, mode: manifest.RopeFreqs, s: S) !mlx.mlx_array {
    const a1 = try axisAngles(freqs, 0, txt_len, mode, s);
    defer _ = mlx.mlx_array_free(a1);
    return concatA(&[_]mlx.mlx_array{ a1, a1, a1 }, 1, s);
}

/// `ang` is ALWAYS f32 — the angle table is an outer product of positions with
/// inverse frequencies, and rounding it before the transcendental is how a
/// rotary embedding goes quietly wrong. Only the resulting cos/sin are cast to
/// the compute dtype, so `applyRope`'s multiply does not promote the whole
/// attention path back to f32.
fn ropeFromAngles(ang: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !Rope {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    var si = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(si);
    try mlx.check(mlx.mlx_cos(&c, ang, s));
    try mlx.check(mlx.mlx_sin(&si, ang, s));
    const cd = try astype(c, dt, s);
    errdefer _ = mlx.mlx_array_free(cd);
    return .{ .cos = cd, .sin = try astype(si, dt, s), .rot = mlx.getShape(ang)[1] };
}

/// Apply interleaved RoPE to `[L, heads, head_dim]`, leaving dims past `rot`
/// untouched. `rotate_half` here is the PAIR form: `(x0, x1) -> (-x1, x0)`.
fn applyRope(x: mlx.mlx_array, r: *const Rope, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(x);
    const l = shp[0];
    const hh = shp[1];
    const hd = shp[2];
    if (r.rot == 0) return contig(x, s);

    const head = try sliceAxis(x, 2, 0, r.rot, s);
    defer _ = mlx.mlx_array_free(head);

    // [L, h, rot] -> [L, h, rot/2, 2] to reach the pairs.
    const pairs = try reshape(head, &[_]c_int{ l, hh, @divExact(r.rot, 2), 2 }, s);
    defer _ = mlx.mlx_array_free(pairs);
    const x0 = try sliceAxis(pairs, 3, 0, 1, s);
    defer _ = mlx.mlx_array_free(x0);
    const x1 = try sliceAxis(pairs, 3, 1, 2, s);
    defer _ = mlx.mlx_array_free(x1);
    var negx1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(negx1);
    try mlx.check(mlx.mlx_negative(&negx1, x1, s));
    // rotate_half: (x0, x1) -> (-x1, x0)
    const rot_pairs = try concatA(&[_]mlx.mlx_array{ negx1, x0 }, 3, s);
    defer _ = mlx.mlx_array_free(rot_pairs);
    const rotated = try reshape(rot_pairs, &[_]c_int{ l, hh, r.rot }, s);
    defer _ = mlx.mlx_array_free(rotated);

    // cos/sin are [L, rot]; broadcast over the head axis.
    const c3 = try reshape(r.cos, &[_]c_int{ l, 1, r.rot }, s);
    defer _ = mlx.mlx_array_free(c3);
    const s3 = try reshape(r.sin, &[_]c_int{ l, 1, r.rot }, s);
    defer _ = mlx.mlx_array_free(s3);
    const a = try mulA(head, c3, s);
    defer _ = mlx.mlx_array_free(a);
    const b = try mulA(rotated, s3, s);
    defer _ = mlx.mlx_array_free(b);
    const out = try addA(a, b, s);
    if (r.rot == hd) return out;
    defer _ = mlx.mlx_array_free(out);
    const tail = try sliceAxis(x, 2, r.rot, hd, s);
    defer _ = mlx.mlx_array_free(tail);
    return concatA(&[_]mlx.mlx_array{ out, tail }, 2, s);
}

// ════════════════════════════════════════════════════════════════════════
// Key scheme — two converters, one set of tensors
// ════════════════════════════════════════════════════════════════════════

/// Two on-disk key layouts carry the same NaDiT weights. This project's own
/// converter (`dit.safetensors`, `justintime47/SeedVR2-3B-MLX-Serve`) nests
/// the txt/vid split as its own path segment (`attn.proj_qkv.txt.weight`,
/// `ada.txt.attn_gate`, `rope.rope.freqs`, `vid_out_ada.out_shift`); the
/// mlx-community 8-bit mirror (`transformer.safetensors`) flattens the split
/// into the tensor name instead (`attn.proj_qkv_txt.weight`,
/// `ada.params_txt.attn_gate`, `rope.freqs`, root-level `out_shift`). Same
/// tensors, same shapes — the MLP names are already identical either way.
/// Detected once from which spelling is actually IN the checkpoint, never
/// assumed from the file it loaded from: a future pack could ship either
/// name under either filename.
const KeyScheme = enum {
    ours,
    mlx_community,

    fn detect(w: *const Weights) KeyScheme {
        return if (w.get("blocks.0.attn.proj_qkv_txt.weight") != null) .mlx_community else .ours;
    }
};

/// `blocks.{layer}.attn.<field>.<stream>` (ours) or
/// `blocks.{layer}.attn.<field>_<stream>` (mlx-community) — the tensor-name
/// PREFIX; the caller appends `.weight`/`.bias`/`.scales`/`.biases`.
///
/// The joiner is not the only difference. A layer past `mm_layers` has ONE
/// attention weight set serving both streams, which our converter names
/// `.all` — and the mirror emits NO `_all` attention tensor at any layer.
/// It materialises that single set under BOTH stream names instead, so `all`
/// resolves to either duplicate (`_vid` by choice; verified byte-identical to
/// `_txt` for all four fields across layers 10-31 of the real int8 pack).
/// Without this the whole shared half of the model is unloadable, which is
/// what `MissingSeedVr2DitWeight` on `blocks.10.attn.proj_qkv_all.weight` was.
/// The MLP and ada groups are NOT affected — the mirror keeps a real `all`
/// there, which is why only the attention family needs the substitution.
fn attnPrefix(a: std.mem.Allocator, layer: u32, comptime field: []const u8, stream: []const u8, scheme: KeyScheme) ![]u8 {
    return switch (scheme) {
        .ours => std.fmt.allocPrint(a, "blocks.{d}.attn." ++ field ++ ".{s}", .{ layer, stream }),
        .mlx_community => blk: {
            const tok: []const u8 = if (std.mem.eql(u8, stream, "all")) "vid" else stream;
            break :blk std.fmt.allocPrint(a, "blocks.{d}.attn." ++ field ++ "_{s}", .{ layer, tok });
        },
    };
}

// ════════════════════════════════════════════════════════════════════════
// Blocks
// ════════════════════════════════════════════════════════════════════════

/// One stream's weights (vid or txt, or the shared `.all` set).
const Stream = struct {
    qkv_wt: DitLinear,
    out_wt: DitLinear,
    out_b: mlx.mlx_array,
    nq_w: mlx.mlx_array,
    nk_w: mlx.mlx_array,
    /// SwiGLU's SiLU'd branch. Null on the plain GELU MLP, which has no gate.
    gate_wt: ?DitLinear,
    up_wt: DitLinear,
    down_wt: DitLinear,
    /// The plain MLP's linears carry biases; SwiGLU's are bias-free.
    up_b: ?mlx.mlx_array,
    down_b: ?mlx.mlx_array,
    ada: [6]mlx.mlx_array, // [layer][kind] flattened: l*3 + kind

    fn load(a: std.mem.Allocator, w: *const Weights, layer: u32, key: []const u8, cfg: Config, scheme: KeyScheme, dt: mlx.mlx_dtype, s: S) !Stream {
        var st: Stream = undefined;

        const qkv_prefix = try attnPrefix(a, layer, "proj_qkv", key, scheme);
        defer a.free(qkv_prefix);
        st.qkv_wt = try loadMixedLinT(w, a, qkv_prefix, cfg.vid_dim, dt, s);

        const out_prefix = try attnPrefix(a, layer, "proj_out", key, scheme);
        defer a.free(out_prefix);
        st.out_wt = try loadMixedLinT(w, a, out_prefix, cfg.innerDim(), dt, s);
        st.out_b = try loadVecAt(w, a, out_prefix, "bias", dt, s);

        const nq_prefix = try attnPrefix(a, layer, "norm_q", key, scheme);
        defer a.free(nq_prefix);
        st.nq_w = try loadVecAt(w, a, nq_prefix, "weight", dt, s);
        const nk_prefix = try attnPrefix(a, layer, "norm_k", key, scheme);
        defer a.free(nk_prefix);
        st.nk_w = try loadVecAt(w, a, nk_prefix, "weight", dt, s);

        // MLP naming is identical across schemes.
        // proj_in_gate is the SiLU'd branch; proj_in is the linear one. The
        // names invite the opposite reading and both shapes match.
        if (cfg.mlp_type == .swiglu) {
            const gate_prefix = try std.fmt.allocPrint(a, "blocks.{d}.mlp.{s}.proj_in_gate", .{ layer, key });
            defer a.free(gate_prefix);
            st.gate_wt = try loadMixedLinT(w, a, gate_prefix, cfg.vid_dim, dt, s);
        } else {
            st.gate_wt = null;
        }
        const up_prefix = try std.fmt.allocPrint(a, "blocks.{d}.mlp.{s}.proj_in", .{ layer, key });
        defer a.free(up_prefix);
        st.up_wt = try loadMixedLinT(w, a, up_prefix, cfg.vid_dim, dt, s);
        const down_prefix = try std.fmt.allocPrint(a, "blocks.{d}.mlp.{s}.proj_out", .{ layer, key });
        defer a.free(down_prefix);
        st.down_wt = try loadMixedLinT(w, a, down_prefix, cfg.mlpHidden(), dt, s);
        // Biases exist only on the plain arm — `bias=True` there, `bias=False`
        // throughout SwiGLU.
        st.up_b = if (cfg.mlp_type == .normal) try loadVecAt(w, a, up_prefix, "bias", dt, s) else null;
        st.down_b = if (cfg.mlp_type == .normal) try loadVecAt(w, a, down_prefix, "bias", dt, s) else null;

        const lnames = [_][]const u8{ "attn", "mlp" };
        const knames = [_][]const u8{ "shift", "scale", "gate" };
        // The ada block's stream token: `txt`/`vid`/`all` unchanged (ours) or
        // `params_txt`/`params_vid`/`params_all` (mlx-community) — the only
        // group where the community mirror renames the token itself rather
        // than changing the joiner.
        const ada_stream = switch (scheme) {
            .ours => try a.dupe(u8, key),
            .mlx_community => try std.fmt.allocPrint(a, "params_{s}", .{key}),
        };
        defer a.free(ada_stream);
        for (lnames, 0..) |ln, li| {
            for (knames, 0..) |kn, ki| {
                st.ada[li * 3 + ki] = try loadVec(w, a, "blocks.{d}.ada.{s}.{s}_{s}", .{ layer, ada_stream, ln, kn }, dt, s);
            }
        }
        return st;
    }

    fn deinit(st: *Stream) void {
        for ([_]*DitLinear{ &st.qkv_wt, &st.out_wt, &st.up_wt, &st.down_wt }) |p| p.deinit();
        if (st.gate_wt) |*g| g.deinit();
        for ([_]*?mlx.mlx_array{ &st.up_b, &st.down_b }) |p|
            if (p.*) |v| {
                _ = mlx.mlx_array_free(v);
            };
        for ([_]*mlx.mlx_array{ &st.out_b, &st.nq_w, &st.nk_w }) |p|
            _ = mlx.mlx_array_free(p.*);
        for (&st.ada) |*p| _ = mlx.mlx_array_free(p.*);
    }

    fn adaOf(st: *const Stream, layer: u32, kind: u32) mlx.mlx_array {
        return st.ada[layer * 3 + kind];
    }

    fn mlp(st: *const Stream, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        // Plain GELU arm (7B): proj_in -> gelu -> proj_out, both with biases.
        // No gate, and the hidden width is `dim * expand_ratio` rather than
        // SwiGLU's two-thirds rounding.
        const gate = st.gate_wt orelse {
            const h = try linT(x, &st.up_wt, st.up_b, s);
            defer _ = mlx.mlx_array_free(h);
            const act = try geluTanh(h, s);
            defer _ = mlx.mlx_array_free(act);
            return linT(act, &st.down_wt, st.down_b, s);
        };
        const g = try linT(x, &gate, null, s);
        defer _ = mlx.mlx_array_free(g);
        const act = try siluA(g, s);
        defer _ = mlx.mlx_array_free(act);
        const u = try linT(x, &st.up_wt, null, s);
        defer _ = mlx.mlx_array_free(u);
        const p = try mulA(act, u, s);
        defer _ = mlx.mlx_array_free(p);
        return linT(p, &st.down_wt, null, s);
    }
};

const Block = struct {
    vid: Stream,
    /// Null when the layer is past `mm_layers` — one shared weight set serves
    /// both streams and `vid` holds it.
    txt: ?Stream,
    rope_freqs: mlx.mlx_array,
    method: win_mod.Method,
    is_last: bool,

    fn deinit(b: *Block) void {
        b.vid.deinit();
        if (b.txt) |*t| t.deinit();
        _ = mlx.mlx_array_free(b.rope_freqs);
    }

    fn txtStream(b: *const Block) *const Stream {
        return if (b.txt) |*t| t else &b.vid;
    }
};

pub const Model = struct {
    cfg: Config,
    vid_in_wt: DitLinear,
    vid_in_b: mlx.mlx_array,
    txt_in_wt: DitLinear,
    txt_in_b: mlx.mlx_array,
    emb_in_wt: [3]DitLinear,
    emb_in_b: [3]mlx.mlx_array,
    blocks: []Block,
    out_norm_w: ?mlx.mlx_array,
    /// All three null when `use_output_ada` is off (7B): the reference builds
    /// them together or not at all.
    out_shift: ?mlx.mlx_array,
    out_scale: ?mlx.mlx_array,
    vid_out_wt: DitLinear,
    vid_out_b: mlx.mlx_array,
    /// The dtype every weight was widened to. `forward` casts its f32 inputs
    /// into it and casts the result back, so callers never see it.
    dtype: mlx.mlx_dtype,
    alloc: std.mem.Allocator,

    pub fn deinit(m: *Model) void {
        m.vid_in_wt.deinit();
        m.txt_in_wt.deinit();
        m.vid_out_wt.deinit();
        for ([_]*mlx.mlx_array{ &m.vid_in_b, &m.txt_in_b, &m.vid_out_b }) |p|
            _ = mlx.mlx_array_free(p.*);
        for ([_]*?mlx.mlx_array{ &m.out_norm_w, &m.out_shift, &m.out_scale }) |p|
            if (p.*) |v| {
                _ = mlx.mlx_array_free(v);
            };
        for (&m.emb_in_wt) |*p| p.deinit();
        for (&m.emb_in_b) |*p| _ = mlx.mlx_array_free(p.*);
        for (m.blocks) |*b| b.deinit();
        m.alloc.free(m.blocks);
    }
};

/// Load at the process-wide compute dtype (`computeDtype`). Serving path.
pub fn load(a: std.mem.Allocator, w: *const Weights, cfg: Config, s: S) !Model {
    return loadAs(a, w, cfg, computeDtype(), s);
}

/// Load at an EXPLICIT dtype. The parity fixtures use this to pin f32: a test
/// that compares against an f32-dumped oracle must not silently start reading
/// its own bf16 rounding back as agreement when the serving default moves.
pub fn loadAs(a: std.mem.Allocator, w: *const Weights, cfg: Config, dt: mlx.mlx_dtype, s: S) !Model {
    var m: Model = undefined;
    m.cfg = cfg;
    m.alloc = a;
    m.dtype = dt;
    const scheme = KeyScheme.detect(w);

    m.vid_in_wt = try loadMixedLinT(w, a, "vid_in.proj", cfg.patchInDim(), dt, s);
    m.vid_in_b = try loadVecAt(w, a, "vid_in.proj", "bias", dt, s);
    m.txt_in_wt = try loadMixedLinT(w, a, "txt_in", cfg.txt_in_dim, dt, s);
    m.txt_in_b = try loadVecAt(w, a, "txt_in", "bias", dt, s);
    const embn = [_][]const u8{ "proj_in", "proj_hid", "proj_out" };
    const emb_in_features = [_]u32{ cfg.sinusoidal_dim, cfg.vid_dim, cfg.vid_dim };
    for (embn, 0..) |n, i| {
        const prefix = try std.fmt.allocPrint(a, "emb_in.{s}", .{n});
        defer a.free(prefix);
        m.emb_in_wt[i] = try loadMixedLinT(w, a, prefix, emb_in_features[i], dt, s);
        m.emb_in_b[i] = try loadVecAt(w, a, prefix, "bias", dt, s);
    }

    m.blocks = try a.alloc(Block, cfg.num_layers);
    var i: u32 = 0;
    while (i < cfg.num_layers) : (i += 1) {
        const branch = manifest.branchForLayer(cfg, i);
        var b: Block = undefined;
        switch (branch) {
            .split => {
                b.vid = try Stream.load(a, w, i, "vid", cfg, scheme, dt, s);
                b.txt = try Stream.load(a, w, i, "txt", cfg, scheme, dt, s);
            },
            .shared => {
                b.vid = try Stream.load(a, w, i, "all", cfg, scheme, dt, s);
                b.txt = null;
            },
        }
        // The inverse-frequency table stays f32 whatever the compute dtype:
        // it is multiplied by positions and fed to cos/sin, and bf16 there is
        // ~3 significant digits of ANGLE.
        b.rope_freqs = switch (scheme) {
            .ours => try loadVec(w, a, "blocks.{d}.attn.rope.rope.freqs", .{i}, mlx.mlx_dtype.float32, s),
            .mlx_community => try loadVec(w, a, "blocks.{d}.attn.rope.freqs", .{i}, mlx.mlx_dtype.float32, s),
        };
        b.method = win_mod.methodForLayer(i);
        b.is_last = manifest.txtSkipsMlp(cfg, i);
        m.blocks[i] = b;
    }

    // The output norm and its modulation are ONE switch in the reference — a
    // single `if use_output_ada` builds all three — and the 7B ships none of
    // them, so a load that reached for `vid_out_norm.weight` here would fail
    // on a checkpoint that is complete.
    if (cfg.use_output_ada) {
        m.out_norm_w = try loadVec(w, a, "vid_out_norm.weight", .{}, dt, s);
        m.out_shift = switch (scheme) {
            .ours => try loadVec(w, a, "vid_out_ada.out_shift", .{}, dt, s),
            .mlx_community => try loadVec(w, a, "out_shift", .{}, dt, s),
        };
        m.out_scale = switch (scheme) {
            .ours => try loadVec(w, a, "vid_out_ada.out_scale", .{}, dt, s),
            .mlx_community => try loadVec(w, a, "out_scale", .{}, dt, s),
        };
    } else {
        m.out_norm_w = null;
        m.out_shift = null;
        m.out_scale = null;
    }
    m.vid_out_wt = try loadMixedLinT(w, a, "vid_out.proj", cfg.vid_dim, dt, s);
    m.vid_out_b = try loadVecAt(w, a, "vid_out.proj", "bias", dt, s);
    return m;
}

/// Sinusoidal timestep embedding, diffusers' `get_timestep_embedding` with
/// `flip_sin_to_cos=False` and `downscale_freq_shift=0`: **sin first**, then
/// cos. Diffusers' own default is the opposite, and swapping them exchanges the
/// halves of the 256-vector. The timestep is on a 0..1000 scale, not 0..1.
fn timestepEmbedding(t: f32, dim: c_int, s: S) !mlx.mlx_array {
    const half = @divExact(dim, 2);
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_arange(&idx, 0.0, @floatFromInt(half), 1.0, mlx.mlx_dtype.float32, s));
    const scale = mlx.mlx_array_new_float(-@log(@as(f32, 10000.0)) / @as(f32, @floatFromInt(half)));
    defer _ = mlx.mlx_array_free(scale);
    const se = try mulA(idx, scale, s);
    defer _ = mlx.mlx_array_free(se);
    var ex = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ex);
    try mlx.check(mlx.mlx_exp(&ex, se, s));
    const tv = mlx.mlx_array_new_float(t);
    defer _ = mlx.mlx_array_free(tv);
    const ang = try mulA(ex, tv, s);
    defer _ = mlx.mlx_array_free(ang);
    var sn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sn);
    var cs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cs);
    try mlx.check(mlx.mlx_sin(&sn, ang, s));
    try mlx.check(mlx.mlx_cos(&cs, ang, s));
    const cat = try concatA(&[_]mlx.mlx_array{ sn, cs }, 0, s);
    defer _ = mlx.mlx_array_free(cat);
    return reshape(cat, &[_]c_int{ 1, dim }, s);
}

fn embedTimestep(m: *const Model, t: f32, s: S) !mlx.mlx_array {
    // The sinusoid is built in f32 — `exp(-log(10000) * i/half) * t` with t up
    // to 1000 spans nine orders of magnitude — and cast only on the way into
    // the first projection.
    const sin32 = try timestepEmbedding(t, @intCast(m.cfg.sinusoidal_dim), s);
    defer _ = mlx.mlx_array_free(sin32);
    const sin = try astype(sin32, m.dtype, s);
    defer _ = mlx.mlx_array_free(sin);
    const a = try linT(sin, &m.emb_in_wt[0], m.emb_in_b[0], s);
    defer _ = mlx.mlx_array_free(a);
    const a2 = try siluA(a, s);
    defer _ = mlx.mlx_array_free(a2);
    const b = try linT(a2, &m.emb_in_wt[1], m.emb_in_b[1], s);
    defer _ = mlx.mlx_array_free(b);
    const b2 = try siluA(b, s);
    defer _ = mlx.mlx_array_free(b2);
    return linT(b2, &m.emb_in_wt[2], m.emb_in_b[2], s);
}

fn idxArray(idx: []const i32, s: S) !mlx.mlx_array {
    _ = s;
    const shp = [_]c_int{@intCast(idx.len)};
    // mlx_array_new_data COPIES shape-worth of bytes, so the slice may die
    // immediately after.
    return mlx.mlx_array_new_data(idx.ptr, &shp, shp.len, mlx.mlx_dtype.int32);
}

/// The window geometry for ONE method on ONE token grid, built once per
/// forward and reused by all sixteen layers that share the method.
///
/// WHY THIS EXISTS. The straightforward loop — gather a window, attend, then
/// `put_along_axis` it back — costs one FULL-SIZE copy of the video stream per
/// window: `put_along_axis` writes a whole new `[grid_tokens, heads, dim]`
/// array each time. At 720p that is 9 copies of 36 MB per layer and 32 layers,
/// ~10 GB of pure copy traffic for a scatter that touches each row once. And
/// the partition itself, plus every window's row-index vector, was rebuilt
/// from scratch on all 32 layers.
///
/// So the whole reverse becomes ONE gather. `gather` lays the windows out
/// end to end (window-major, the reference's emission order), `scatter` is its
/// inverse — for each grid row, the LAST gathered slot that covers it — and
/// `take(concat(window_outputs), scatter)` reproduces `put_along_axis`'s
/// last-write-wins exactly, including for a hypothetical overlapping layout
/// that the current partition never actually produces.
const WindowPlan = struct {
    alloc: std.mem.Allocator,
    windows: []win_mod.Window,
    /// `offsets[i]..offsets[i+1]` is window `i`'s span in gathered order.
    offsets: []u32,
    /// `[total_window_tokens]` — which grid row each gathered slot came from.
    gather: mlx.mlx_array,
    /// `[grid_tokens]` — which gathered slot writes each grid row.
    scatter: mlx.mlx_array,
    /// `extents[extent_of[i]]` is window `i`'s `(t,h,w)`. Windows sharing an
    /// extent share a RoPE table, and on a 3x3 split there are four distinct
    /// extents for nine windows.
    extent_of: []usize,
    extents: [][3]u32,

    fn build(a: std.mem.Allocator, grid: [3]u32, method: win_mod.Method) !WindowPlan {
        const windows = try win_mod.partition(a, grid, NUM_WINDOWS, method);
        errdefer a.free(windows);

        const total: usize = @intCast(win_mod.totalTokens(windows));
        const grid_tokens: usize = @as(usize, grid[0]) * grid[1] * grid[2];

        const offsets = try a.alloc(u32, windows.len + 1);
        errdefer a.free(offsets);
        const gather_rows = try a.alloc(i32, total);
        defer a.free(gather_rows);
        // Never covered stays -1 so a hole is a loud out-of-range gather
        // rather than a silent row of window 0's output. The partition covers
        // every token by construction (pinned in seedvr2_window.zig), so this
        // is a tripwire, not a fallback.
        const scatter_rows = try a.alloc(i32, grid_tokens);
        defer a.free(scatter_rows);
        @memset(scatter_rows, -1);

        const extent_of = try a.alloc(usize, windows.len);
        errdefer a.free(extent_of);
        var extents: std.ArrayList([3]u32) = .empty;
        errdefer extents.deinit(a);

        var k: usize = 0;
        for (windows, 0..) |win, wi| {
            offsets[wi] = @intCast(k);
            const ext = win.extent();
            extent_of[wi] = for (extents.items, 0..) |e, ei| {
                if (std.meta.eql(e, ext)) break ei;
            } else blk: {
                try extents.append(a, ext);
                break :blk extents.items.len - 1;
            };
            var t = win.t0;
            while (t < win.t1) : (t += 1) {
                var h = win.h0;
                while (h < win.h1) : (h += 1) {
                    var w = win.w0;
                    while (w < win.w1) : (w += 1) {
                        const row: i32 = @intCast((t * grid[1] + h) * grid[2] + w);
                        gather_rows[k] = row;
                        // LAST write wins, matching put_along_axis and the
                        // reference's index_select-based reverse.
                        scatter_rows[@intCast(row)] = @intCast(k);
                        k += 1;
                    }
                }
            }
        }
        offsets[windows.len] = @intCast(k);
        std.debug.assert(k == total);
        for (scatter_rows) |r| if (r < 0) return error.SeedVr2WindowHole;

        var plan: WindowPlan = .{
            .alloc = a,
            .windows = windows,
            .offsets = offsets,
            .gather = try idxArray(gather_rows, undefined),
            .scatter = undefined,
            .extent_of = extent_of,
            .extents = try extents.toOwnedSlice(a),
        };
        errdefer _ = mlx.mlx_array_free(plan.gather);
        plan.scatter = try idxArray(scatter_rows, undefined);
        return plan;
    }

    fn deinit(p: *WindowPlan) void {
        _ = mlx.mlx_array_free(p.gather);
        _ = mlx.mlx_array_free(p.scatter);
        p.alloc.free(p.windows);
        p.alloc.free(p.offsets);
        p.alloc.free(p.extent_of);
        p.alloc.free(p.extents);
    }
};

/// The two plans a forward needs. Every EVEN layer is `.plain` and every ODD
/// one `.shifted`, so two plans serve all 32.
const WindowPlans = struct {
    plain: WindowPlan,
    shifted: WindowPlan,

    fn build(a: std.mem.Allocator, grid: [3]u32) !WindowPlans {
        var plain = try WindowPlan.build(a, grid, .plain);
        errdefer plain.deinit();
        return .{ .plain = plain, .shifted = try WindowPlan.build(a, grid, .shifted) };
    }

    fn deinit(p: *WindowPlans) void {
        p.plain.deinit();
        p.shifted.deinit();
    }

    fn for_(p: *const WindowPlans, method: win_mod.Method) *const WindowPlan {
        return switch (method) {
            .plain => &p.plain,
            .shifted => &p.shifted,
        };
    }
};

/// Scaled dot-product attention over one sequence, `[L, heads, dim]` in and out.
///
/// Q/K/V are ROUNDED TO BF16 first. That is not an optimisation — the reference
/// calls `.bfloat16()` on all three before handing them to `flash_attn_varlen_func`
/// (flash-attn takes no f32), so bf16 is the precision the checkpoint is
/// actually evaluated at. Computing in f32 here is *more* accurate and lands
/// ~4e-3 relative away from the reference, which is exactly bf16's resolution
/// and reads like a bug for a whole afternoon. Kill switch for A/Bs:
/// MLX_SERVE_SEEDVR2_ATTN_F32=1.
fn attend(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, s: S) !mlx.mlx_array {
    if (!attnF32()) {
        const qb = try astype(q, mlx.mlx_dtype.bfloat16, s);
        defer _ = mlx.mlx_array_free(qb);
        const kb = try astype(k, mlx.mlx_dtype.bfloat16, s);
        defer _ = mlx.mlx_array_free(kb);
        const vb = try astype(v, mlx.mlx_dtype.bfloat16, s);
        defer _ = mlx.mlx_array_free(vb);
        const q2 = try astype(qb, mlx.mlx_dtype.float32, s);
        defer _ = mlx.mlx_array_free(q2);
        const k2 = try astype(kb, mlx.mlx_dtype.float32, s);
        defer _ = mlx.mlx_array_free(k2);
        const v2 = try astype(vb, mlx.mlx_dtype.float32, s);
        defer _ = mlx.mlx_array_free(v2);
        const o = try attendF32(q2, k2, v2, s);
        defer _ = mlx.mlx_array_free(o);
        // The OUTPUT is bf16 too. flash_attn returns the dtype it was given, and
        // the reference's `.type_as(vid_q)` upcasts an ALREADY-ROUNDED result —
        // it does not recover precision. Rounding only the inputs leaves a
        // uniform ~4e-3 (bf16 epsilon, 2^-8) that reads like a formula bug.
        const ob = try astype(o, mlx.mlx_dtype.bfloat16, s);
        defer _ = mlx.mlx_array_free(ob);
        return astype(ob, mlx.mlx_dtype.float32, s);
    }
    return attendF32(q, k, v, s);
}

fn attnF32() bool {
    const v = std.c.getenv("MLX_SERVE_SEEDVR2_ATTN_F32") orelse return false;
    return std.mem.span(v).len > 0 and std.mem.span(v)[0] == '1';
}

fn attendF32(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(q);
    const hd = shp[2];
    // [L, h, d] -> [h, L, d]
    const qt = try transpose(q, &[_]c_int{ 1, 0, 2 }, s);
    defer _ = mlx.mlx_array_free(qt);
    const kt = try transpose(k, &[_]c_int{ 1, 2, 0 }, s);
    defer _ = mlx.mlx_array_free(kt);
    const vt = try transpose(v, &[_]c_int{ 1, 0, 2 }, s);
    defer _ = mlx.mlx_array_free(vt);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_matmul(&sc, qt, kt, s));
    const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(hd))));
    defer _ = mlx.mlx_array_free(scale);
    const scs = try mulA(sc, scale, s);
    defer _ = mlx.mlx_array_free(scs);
    var pr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pr);
    try mlx.check(mlx.mlx_softmax_axis(&pr, scs, -1, true, s));
    var ctx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ctx);
    try mlx.check(mlx.mlx_matmul(&ctx, pr, vt, s));
    return transpose(ctx, &[_]c_int{ 1, 0, 2 }, s);
}

/// Split a `[L, 3*heads*dim]` qkv projection into three `[L, heads, dim]`.
fn splitQkv(x: mlx.mlx_array, heads: c_int, head_dim: c_int, out: *[3]mlx.mlx_array, s: S) !void {
    const l = mlx.getShape(x)[0];
    // The reference reads `l (o h d)` with o SLOWEST: q, then k, then v.
    const r = try reshape(x, &[_]c_int{ l, 3, heads, head_dim }, s);
    defer _ = mlx.mlx_array_free(r);
    for (0..3) |i| {
        const sl = try sliceAxis(r, 1, @intCast(i), @intCast(i + 1), s);
        defer _ = mlx.mlx_array_free(sl);
        out[i] = try reshape(sl, &[_]c_int{ l, heads, head_dim }, s);
    }
}

const AttnOut = struct { vid: mlx.mlx_array, txt: mlx.mlx_array };

/// Windowed multi-modal attention.
///
/// Every window attends over `[vid_window_tokens, ALL text tokens]`. The text
/// therefore participates `num_windows` times and its outputs are MEAN-POOLED
/// back to one copy — `unconcat_coalesce` in the reference. Taking a single
/// window's copy instead produces output that looks fine and is wrong.
fn attention(
    b: *const Block,
    a: std.mem.Allocator,
    vid: mlx.mlx_array,
    txt: mlx.mlx_array,
    plan: *const WindowPlan,
    cfg: Config,
    s: S,
) !AttnOut {
    const heads: c_int = @intCast(cfg.heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const txt_len = mlx.getShape(txt)[0];
    const dt = mlx.mlx_array_dtype(vid);

    const vs = b.vid;
    const ts = b.txtStream();

    const vqkv = try linT(vid, &vs.qkv_wt, null, s);
    defer _ = mlx.mlx_array_free(vqkv);
    const tqkv = try linT(txt, &ts.qkv_wt, null, s);
    defer _ = mlx.mlx_array_free(tqkv);

    var vsplit: [3]mlx.mlx_array = undefined;
    try splitQkv(vqkv, heads, hd, &vsplit, s);
    defer for (&vsplit) |p| {
        _ = mlx.mlx_array_free(p);
    };
    var tsplit: [3]mlx.mlx_array = undefined;
    try splitQkv(tqkv, heads, hd, &tsplit, s);
    defer for (&tsplit) |p| {
        _ = mlx.mlx_array_free(p);
    };

    // qk-norm is per HEAD_DIM, applied before rope.
    const vq = try rmsNorm(vsplit[0], vs.nq_w, 1e-5, s);
    defer _ = mlx.mlx_array_free(vq);
    const vk = try rmsNorm(vsplit[1], vs.nk_w, 1e-5, s);
    defer _ = mlx.mlx_array_free(vk);
    const tq0 = try rmsNorm(tsplit[0], ts.nq_w, 1e-5, s);
    defer _ = mlx.mlx_array_free(tq0);
    const tk0 = try rmsNorm(tsplit[1], ts.nk_w, 1e-5, s);
    defer _ = mlx.mlx_array_free(tk0);

    // Text rope is the same in every window (window-local positions restart),
    // so it is built once — and on the 7B it is not built at all: with
    // `rope_on_text` off the reference takes the video-only rope path and the
    // text q/k go into attention unrotated.
    const tq = if (cfg.rope_on_text) blk: {
        const tang = try textAngles(b.rope_freqs, txt_len, cfg.rope_freqs_for, s);
        defer _ = mlx.mlx_array_free(tang);
        var trope = try ropeFromAngles(tang, dt, s);
        defer trope.deinit();
        const q = try applyRope(tq0, &trope, s);
        errdefer _ = mlx.mlx_array_free(q);
        break :blk q;
    } else try ownArray(tq0);
    defer _ = mlx.mlx_array_free(tq);
    const tk = if (cfg.rope_on_text) blk: {
        const tang = try textAngles(b.rope_freqs, txt_len, cfg.rope_freqs_for, s);
        defer _ = mlx.mlx_array_free(tang);
        var trope = try ropeFromAngles(tang, dt, s);
        defer trope.deinit();
        break :blk try applyRope(tk0, &trope, s);
    } else try ownArray(tk0);
    defer _ = mlx.mlx_array_free(tk);

    // ONE RoPE table per DISTINCT window extent, not per window. A 3x3 split
    // has nine windows and four extents (interior, right edge, bottom edge,
    // corner), and the tables are pure functions of the extent.
    const ropes = try a.alloc(Rope, plan.extents.len);
    defer a.free(ropes);
    var built: usize = 0;
    defer for (ropes[0..built]) |*r| r.deinit();
    for (plan.extents) |ext| {
        const rope_txt_off: c_int = if (cfg.rope_on_text) txt_len else 0;
        const vang = try videoAngles(b.rope_freqs, @intCast(ext[0]), @intCast(ext[1]), @intCast(ext[2]), rope_txt_off, cfg.rope_freqs_for, a, s);
        defer _ = mlx.mlx_array_free(vang);
        ropes[built] = try ropeFromAngles(vang, dt, s);
        built += 1;
    }

    // Gather EVERY window's rows in one dispatch instead of three per window.
    const gq = try takeRows(vq, plan.gather, s);
    defer _ = mlx.mlx_array_free(gq);
    const gk = try takeRows(vk, plan.gather, s);
    defer _ = mlx.mlx_array_free(gk);
    const gv = try takeRows(vsplit[2], plan.gather, s);
    defer _ = mlx.mlx_array_free(gv);

    const outs = try a.alloc(mlx.mlx_array, plan.windows.len);
    defer a.free(outs);
    var n_out: usize = 0;
    defer for (outs[0..n_out]) |o| {
        _ = mlx.mlx_array_free(o);
    };

    var txt_acc = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&txt_acc, &[_]c_int{ txt_len, heads, hd }, 3, dt, s));
    errdefer _ = mlx.mlx_array_free(txt_acc);

    for (plan.windows, 0..) |_, wi| {
        const lo: c_int = @intCast(plan.offsets[wi]);
        const hi: c_int = @intCast(plan.offsets[wi + 1]);
        const wl = hi - lo;
        const vrope = &ropes[plan.extent_of[wi]];

        const wq0 = try sliceAxis(gq, 0, lo, hi, s);
        defer _ = mlx.mlx_array_free(wq0);
        const wk0 = try sliceAxis(gk, 0, lo, hi, s);
        defer _ = mlx.mlx_array_free(wk0);
        const wv = try sliceAxis(gv, 0, lo, hi, s);
        defer _ = mlx.mlx_array_free(wv);

        const wq = try applyRope(wq0, vrope, s);
        defer _ = mlx.mlx_array_free(wq);
        const wk = try applyRope(wk0, vrope, s);
        defer _ = mlx.mlx_array_free(wk);

        const cq = try concatA(&[_]mlx.mlx_array{ wq, tq }, 0, s);
        defer _ = mlx.mlx_array_free(cq);
        const ck = try concatA(&[_]mlx.mlx_array{ wk, tk }, 0, s);
        defer _ = mlx.mlx_array_free(ck);
        const cv = try concatA(&[_]mlx.mlx_array{ wv, tsplit[2] }, 0, s);
        defer _ = mlx.mlx_array_free(cv);

        const o = try attend(cq, ck, cv, s);
        defer _ = mlx.mlx_array_free(o);
        outs[n_out] = try sliceAxis(o, 0, 0, wl, s);
        n_out += 1;
        const ot = try sliceAxis(o, 0, wl, wl + txt_len, s);
        defer _ = mlx.mlx_array_free(ot);

        const na = try addA(txt_acc, ot, s);
        _ = mlx.mlx_array_free(txt_acc);
        txt_acc = na;
    }

    // The reverse, in ONE gather: lay the windows out end to end in emission
    // order and take each grid row from the slot that owns it. The old form
    // rebuilt the whole `[grid_tokens, heads, dim]` array once PER WINDOW.
    const stacked = try concatA(outs[0..n_out], 0, s);
    defer _ = mlx.mlx_array_free(stacked);
    const vid_out = try takeRows(stacked, plan.scatter, s);
    defer _ = mlx.mlx_array_free(vid_out);

    // MEAN over windows — not the last copy, not the first.
    const inv = mlx.mlx_array_new_float(1.0 / @as(f32, @floatFromInt(plan.windows.len)));
    defer _ = mlx.mlx_array_free(inv);
    const txt_mean = try mulA(txt_acc, inv, s);
    _ = mlx.mlx_array_free(txt_acc);
    defer _ = mlx.mlx_array_free(txt_mean);

    const total_v: c_int = mlx.getShape(vid_out)[0];
    const vflat = try reshape(vid_out, &[_]c_int{ total_v, heads * hd }, s);
    defer _ = mlx.mlx_array_free(vflat);
    const tflat = try reshape(txt_mean, &[_]c_int{ txt_len, heads * hd }, s);
    defer _ = mlx.mlx_array_free(tflat);
    return .{
        .vid = try linT(vflat, &vs.out_wt, vs.out_b, s),
        .txt = try linT(tflat, &ts.out_wt, ts.out_b, s),
    };
}

fn runBlock(
    b: *const Block,
    a: std.mem.Allocator,
    vid_in: mlx.mlx_array,
    txt_in: mlx.mlx_array,
    m: *const Modulation,
    plan: *const WindowPlan,
    cfg: Config,
    s: S,
) !AttnOut {
    const vs = b.vid;
    const ts = b.txtStream();

    const vn = try rmsNorm(vid_in, null, 1e-5, s);
    defer _ = mlx.mlx_array_free(vn);
    const tn = try rmsNorm(txt_in, null, 1e-5, s);
    defer _ = mlx.mlx_array_free(tn);
    const vm = try adaIn(vn, m, LAYER_ATTN, vs.adaOf(LAYER_ATTN, KIND_SHIFT), vs.adaOf(LAYER_ATTN, KIND_SCALE), s);
    defer _ = mlx.mlx_array_free(vm);
    // `self.ada` carries vid_only=is_last, so in the FINAL block the text
    // stream receives NO modulation at all — not the attn shift/scale here, and
    // not the attn gate below. Only `attn_norm` still applies to it (that one
    // module is built without vid_only). Modulating txt here anyway leaves the
    // magnitudes plausible and the direction wrong: cos 0.9998 against 1.000000
    // on every other block.
    const tm = if (b.is_last) try contig(tn, s) else try adaIn(tn, m, LAYER_ATTN, ts.adaOf(LAYER_ATTN, KIND_SHIFT), ts.adaOf(LAYER_ATTN, KIND_SCALE), s);
    defer _ = mlx.mlx_array_free(tm);

    const at = try attention(b, a, vm, tm, plan, cfg, s);
    defer _ = mlx.mlx_array_free(at.vid);
    defer _ = mlx.mlx_array_free(at.txt);

    const vg = try adaOut(at.vid, m, LAYER_ATTN, vs.adaOf(LAYER_ATTN, KIND_GATE), s);
    defer _ = mlx.mlx_array_free(vg);
    const tg = if (b.is_last) try contig(at.txt, s) else try adaOut(at.txt, m, LAYER_ATTN, ts.adaOf(LAYER_ATTN, KIND_GATE), s);
    defer _ = mlx.mlx_array_free(tg);
    const v1 = try addA(vg, vid_in, s);
    const t1 = try addA(tg, txt_in, s);
    errdefer _ = mlx.mlx_array_free(v1);
    errdefer _ = mlx.mlx_array_free(t1);

    // --- MLP ---
    const vn2 = try rmsNorm(v1, null, 1e-5, s);
    defer _ = mlx.mlx_array_free(vn2);
    const vm2 = try adaIn(vn2, m, LAYER_MLP, vs.adaOf(LAYER_MLP, KIND_SHIFT), vs.adaOf(LAYER_MLP, KIND_SCALE), s);
    defer _ = mlx.mlx_array_free(vm2);
    const vmlp = try vs.mlp(vm2, s);
    defer _ = mlx.mlx_array_free(vmlp);
    const vgate = try adaOut(vmlp, m, LAYER_MLP, vs.adaOf(LAYER_MLP, KIND_GATE), s);
    defer _ = mlx.mlx_array_free(vgate);
    const v2 = try addA(vgate, v1, s);
    _ = mlx.mlx_array_free(v1);

    // THE LAST BLOCK DOUBLES ITS TEXT OUTPUT, and that is not a typo.
    //
    // `is_last` sets `vid_only=True` on mlp_norm / mlp / ada, and `MMModule`
    // with vid_only PASSES TXT THROUGH UNCHANGED rather than skipping the
    // stage. The block's residual add is then unconditional:
    //     vid_mlp, txt_mlp = (vid_mlp + vid_attn), (txt_mlp + txt_attn)
    // and since `txt_mlp` IS `txt_attn` at that point, the text stream is added
    // to itself. Returning t1 instead of 2*t1 shows up as a max relative error
    // of exactly 0.5 against the reference.
    //
    // Nothing downstream consumes the final block's text — `forward` frees it —
    // so this changes no output. It is reproduced because a port that quietly
    // disagrees with the reference somewhere harmless is a port you cannot use
    // to localise a bug somewhere harmful.
    if (b.is_last) {
        const t2last = try addA(t1, t1, s);
        _ = mlx.mlx_array_free(t1);
        return .{ .vid = v2, .txt = t2last };
    }

    const tn2 = try rmsNorm(t1, null, 1e-5, s);
    defer _ = mlx.mlx_array_free(tn2);
    const tm2 = try adaIn(tn2, m, LAYER_MLP, ts.adaOf(LAYER_MLP, KIND_SHIFT), ts.adaOf(LAYER_MLP, KIND_SCALE), s);
    defer _ = mlx.mlx_array_free(tm2);
    const tmlp = try ts.mlp(tm2, s);
    defer _ = mlx.mlx_array_free(tmlp);
    const tgate = try adaOut(tmlp, m, LAYER_MLP, ts.adaOf(LAYER_MLP, KIND_GATE), s);
    defer _ = mlx.mlx_array_free(tgate);
    const t2 = try addA(tgate, t1, s);
    _ = mlx.mlx_array_free(t1);
    return .{ .vid = v2, .txt = t2 };
}

/// Patchify `[T*H*W, C]` over a `(T,H,W)` grid into `[T*(H/2)*(W/2), 4*C]`.
/// Channel order inside a patch is `(h w c)` — c fastest, then w, then h.
fn patchify(x: mlx.mlx_array, grid: [3]u32, s: S) !mlx.mlx_array {
    const c = mlx.getShape(x)[1];
    const t: c_int = @intCast(grid[0]);
    const h: c_int = @intCast(grid[1]);
    const w: c_int = @intCast(grid[2]);
    const g = try reshape(x, &[_]c_int{ t, @divExact(h, 2), 2, @divExact(w, 2), 2, c }, s);
    defer _ = mlx.mlx_array_free(g);
    // (T, H/2, h, W/2, w, C) -> (T, H/2, W/2, h, w, C)
    const p = try transpose(g, &[_]c_int{ 0, 1, 3, 2, 4, 5 }, s);
    defer _ = mlx.mlx_array_free(p);
    const pc = try contig(p, s);
    defer _ = mlx.mlx_array_free(pc);
    return reshape(pc, &[_]c_int{ t * @divExact(h, 2) * @divExact(w, 2), 4 * c }, s);
}

fn unpatchify(x: mlx.mlx_array, grid: [3]u32, out_c: c_int, s: S) !mlx.mlx_array {
    const t: c_int = @intCast(grid[0]);
    const h2: c_int = @intCast(grid[1]);
    const w2: c_int = @intCast(grid[2]);
    const g = try reshape(x, &[_]c_int{ t, h2, w2, 2, 2, out_c }, s);
    defer _ = mlx.mlx_array_free(g);
    const p = try transpose(g, &[_]c_int{ 0, 1, 3, 2, 4, 5 }, s);
    defer _ = mlx.mlx_array_free(p);
    const pc = try contig(p, s);
    defer _ = mlx.mlx_array_free(pc);
    return reshape(pc, &[_]c_int{ t * h2 * 2 * w2 * 2, out_c }, s);
}

/// One full NaDiT forward.
///
/// `vid` is `[T*H*W, vid_in_channels]` over the LATENT grid (pre-patchify);
/// `txt` is `[Lt, txt_in_dim]`. Returns `[T*H*W, vid_out_channels]` on the same
/// latent grid.
pub fn forward(
    m: *const Model,
    a: std.mem.Allocator,
    vid: mlx.mlx_array,
    txt: mlx.mlx_array,
    grid: [3]u32,
    timestep: f32,
    s: S,
) !mlx.mlx_array {
    const patched32 = try patchify(vid, grid, s);
    defer _ = mlx.mlx_array_free(patched32);
    const patched = try astype(patched32, m.dtype, s);
    defer _ = mlx.mlx_array_free(patched);
    const txt_dt = try astype(txt, m.dtype, s);
    defer _ = mlx.mlx_array_free(txt_dt);
    var cur_v = try linT(patched, &m.vid_in_wt, m.vid_in_b, s);
    var cur_t = try linT(txt_dt, &m.txt_in_wt, m.txt_in_b, s);

    const tok_grid: [3]u32 = .{ grid[0], grid[1] / 2, grid[2] / 2 };

    const emb = try embedTimestep(m, timestep, s);
    defer _ = mlx.mlx_array_free(emb);
    var mod = try Modulation.init(emb, @intCast(m.cfg.vid_dim), s);
    defer mod.deinit();

    // The window geometry is a function of the token grid and the method, and
    // NOTHING else — so two plans serve all 32 layers instead of 32 partitions
    // and 32 sets of index vectors.
    var plans = try WindowPlans.build(a, tok_grid);
    defer plans.deinit();

    for (m.blocks) |*b| {
        const out = try runBlock(b, a, cur_v, cur_t, &mod, plans.for_(b.method), m.cfg, s);
        _ = mlx.mlx_array_free(cur_v);
        _ = mlx.mlx_array_free(cur_t);
        cur_v = out.vid;
        cur_t = out.txt;
    }
    _ = mlx.mlx_array_free(cur_t);
    defer _ = mlx.mlx_array_free(cur_v);

    // `use_output_ada` off (7B): the reference goes straight from the last
    // block to `vid_out`, with no norm and no modulation in between.
    const head_in: mlx.mlx_array = if (m.out_norm_w) |nw| blk: {
        const n = try rmsNorm(cur_v, nw, 1e-5, s);
        defer _ = mlx.mlx_array_free(n);
        // THE CACHE COLLISION (docs/seedvr2-arch.md §1.3.1): vid_out_ada reuses
        // the ATTN modulation slot, because AdaSingle's memo key for
        // layers=["out"] is `emb_repeat_0_vid` — exactly what the blocks' attn
        // ada already stored. Its own 1-layer view is malformed and never
        // evaluated.
        break :blk try adaIn(n, &mod, LAYER_ATTN, m.out_shift.?, m.out_scale.?, s);
    } else try ownArray(cur_v);
    defer _ = mlx.mlx_array_free(head_in);
    const proj_dt = try linT(head_in, &m.vid_out_wt, m.vid_out_b, s);
    defer _ = mlx.mlx_array_free(proj_dt);
    // Back to f32 at the boundary: the sampler's `x_0 = noise - pred` and the
    // VAE downstream are f32, and a bf16 prediction subtracted from f32 noise
    // would promote silently anyway — better to say so here.
    const proj = try astype(proj_dt, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(proj);
    return unpatchify(proj, tok_grid, @intCast(m.cfg.vid_out_channels), s);
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;


/// The tiny fixture's geometry (tests/dump_seedvr2_fixtures.py dit).
fn tinyConfig() Config {
    return .{
        .vid_dim = 64,
        .txt_in_dim = 80,
        .heads = 2,
        .head_dim = 32,
        .num_layers = 4,
        .mm_layers = 2,
        .expand_ratio = 4,
        .mlp_multiple_of = 256,
        .vid_in_channels = 33,
        .vid_out_channels = 16,
        .sinusoidal_dim = 256,
        .rope_dim = 24,
    };
}

fn fxGet(fx: *const Weights, name: []const u8, s: S) !mlx.mlx_array {
    const raw = try ownWeight(fx, name);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, mlx.mlx_dtype.float32, s);
}

fn cosOf(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: S) !f32 {
    const ab = try mulA(a_arr, b_arr, s);
    defer _ = mlx.mlx_array_free(ab);
    const aa = try mulA(a_arr, a_arr, s);
    defer _ = mlx.mlx_array_free(aa);
    const bb = try mulA(b_arr, b_arr, s);
    defer _ = mlx.mlx_array_free(bb);
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    var na = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(na);
    var nb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nb);
    try mlx.check(mlx.mlx_sum(&d, ab, false, s));
    try mlx.check(mlx.mlx_sum(&na, aa, false, s));
    try mlx.check(mlx.mlx_sum(&nb, bb, false, s));
    try mlx.check(mlx.mlx_array_eval(d));
    try mlx.check(mlx.mlx_array_eval(na));
    try mlx.check(mlx.mlx_array_eval(nb));
    var dv: f32 = 0;
    var av: f32 = 0;
    var bv: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&dv, d));
    try mlx.check(mlx.mlx_array_item_float32(&av, na));
    try mlx.check(mlx.mlx_array_item_float32(&bv, nb));
    return dv / (@sqrt(av) * @sqrt(bv) + 1e-12);
}

fn maxAbsDiff(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: S) !struct { diff: f32, scale: f32 } {
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, a_arr, b_arr, s));
    var ad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ad);
    try mlx.check(mlx.mlx_abs(&ad, d, s));
    var md = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(md);
    try mlx.check(mlx.mlx_max(&md, ad, false, s));
    var ab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ab);
    try mlx.check(mlx.mlx_abs(&ab, b_arr, s));
    var mb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mb);
    try mlx.check(mlx.mlx_max(&mb, ab, false, s));
    try mlx.check(mlx.mlx_array_eval(md));
    try mlx.check(mlx.mlx_array_eval(mb));
    var dv: f32 = 0;
    var bv: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&dv, md));
    try mlx.check(mlx.mlx_array_item_float32(&bv, mb));
    return .{ .diff = dv, .scale = bv };
}

/// Compare one stage against the reference.
///
/// Reports max-abs error RELATIVE TO the reference's own magnitude beside the
/// cosine, because the two answer different questions: cosine is blind to a
/// uniform scale, and on a small tensor it is also noisy in a way that looks
/// like a bug. A stage whose cosine dips but whose relative error stays at the
/// f32 floor is arithmetic ordering, not a wrong formula.
fn expectStage(tag: []const u8, got: mlx.mlx_array, fx: *const Weights, name: []const u8, s: S) !void {
    const ref = try fxGet(fx, name, s);
    defer _ = mlx.mlx_array_free(ref);
    const gs = mlx.getShape(got);
    const rs = mlx.getShape(ref);
    testing.expectEqualSlices(c_int, rs, gs) catch |err| {
        std.debug.print("{s}: shape {any} != reference {any}\n", .{ tag, gs, rs });
        return err;
    };
    const c = try cosOf(got, ref, s);
    const e = try maxAbsDiff(got, ref, s);
    const rel = e.diff / (e.scale + 1e-12);
    std.debug.print("[seedvr2-dit] {s}: cos={d:.6} max_rel_err={e:.3}\n", .{ tag, c, rel });

    // TWO BARS, and the loose one is loose for a measured reason.
    //
    // COSINE is the tight bar and the one that catches structural bugs: the
    // three breaks this test was validated against scored 0.9964 (text summed
    // instead of mean-pooled), 0.544 (SiLU branch swapped) and -0.228
    // (vid_out_ada on the wrong modulation slot).
    //
    // MAX RELATIVE ERROR cannot be tightened past bf16 resolution. The
    // reference runs attention in bf16 — `.bfloat16()` on q/k/v, and the result
    // comes back bf16 because flash-attn returns the dtype it was given — so
    // its activations sit on the bf16 grid. Any difference in reduction order
    // before that rounding lands on a neighbouring grid point, and one bf16 ulp
    // IS 2^-8 = 3.9e-3. Measured here: 2.2e-3 at cos 1.000000. A bar below that
    // would be a test that can only be passed by luck.
    testing.expect(rel < 8.0e-3) catch |err| {
        std.debug.print("{s}: max relative error {e:.3} exceeds 2 bf16 ulp (cos {d:.6})\n", .{ tag, rel, c });
        return err;
    };
    testing.expect(c > 0.9999) catch |err| {
        std.debug.print("{s}: cosine {d:.6} below 0.9999\n", .{ tag, c });
        return err;
    };
}

test "seedvr2 dit: tiny NaDiT matches the reference stage by stage" {
    // PARITY against a small random-weight NaDiT built and run by the
    // reference (tests/dump_seedvr2_fixtures.py dit). Enable with
    //   -Dseedvr2-fixtures=tests/fixtures/seedvr2
    //
    // Staged deliberately: a final-output-only check tells you the port is
    // wrong and nothing else. With per-block references the first red stage
    // names the layer, and the split/shared boundary sits at layer 2 here so
    // "block 0 green, block 2 red" localises to the naming rather than to the
    // maths.
    const a = testing.allocator;
    const s = mlx.gpuStream();

    // COMMITTED fixture, deliberately not env-gated. It carries its own random
    // weights, so it needs no checkpoint and this test ALWAYS runs — the same
    // reasoning .gitignore gives for src/fixtures/minimax_h3_dit.safetensors.
    // Regenerate with:
    //   tests/dump_seedvr2_fixtures.py dit --seedvr-src <SeedVR> --out src/fixtures/
    var fx = try model_mod.loadWeightsSingleFile(a, "src/fixtures/seedvr2_dit_tiny.safetensors");
    defer fx.deinit();

    const cfg = tinyConfig();
    var m = try loadAs(a, &fx, cfg, mlx.mlx_dtype.float32, s);
    defer m.deinit();

    const grid: [3]u32 = .{ 2, 48, 48 };
    const tok_grid: [3]u32 = .{ 2, 24, 24 };
    const timestep: f32 = 734.0;

    const vid0 = try fxGet(&fx, "in_vid", s);
    defer _ = mlx.mlx_array_free(vid0);
    const txt0 = try fxGet(&fx, "in_txt", s);
    defer _ = mlx.mlx_array_free(txt0);

    // Stage 1: patchify + vid_in.
    const patched = try patchify(vid0, grid, s);
    defer _ = mlx.mlx_array_free(patched);
    var cur_v = try linT(patched, &m.vid_in_wt, m.vid_in_b, s);
    defer _ = mlx.mlx_array_free(cur_v);
    try expectStage("vid_in", cur_v, &fx, "act_vid_in", s);

    // Stage 2: the timestep embedding. Its sin/cos ordering is a coin flip
    // that only this stage can catch cleanly.
    const emb = try embedTimestep(&m, timestep, s);
    defer _ = mlx.mlx_array_free(emb);
    try expectStage("emb", emb, &fx, "act_emb", s);

    var mod = try Modulation.init(emb, @intCast(cfg.vid_dim), s);
    defer mod.deinit();

    var cur_t = try linT(txt0, &m.txt_in_wt, m.txt_in_b, s);
    defer _ = mlx.mlx_array_free(cur_t);

    var plans = try WindowPlans.build(a, tok_grid);
    defer plans.deinit();

    // Stage 3: every block, in order.
    for (m.blocks, 0..) |*b, i| {
        const out = try runBlock(b, a, cur_v, cur_t, &mod, plans.for_(b.method), cfg, s);
        _ = mlx.mlx_array_free(cur_v);
        _ = mlx.mlx_array_free(cur_t);
        cur_v = out.vid;
        cur_t = out.txt;
        const vn = try std.fmt.allocPrint(a, "act_block{d}_vid", .{i});
        defer a.free(vn);
        const tn = try std.fmt.allocPrint(a, "act_block{d}_txt", .{i});
        defer a.free(tn);
        try expectStage(vn, cur_v, &fx, vn, s);
        try expectStage(tn, cur_t, &fx, tn, s);
    }

    // Stage 4: the whole forward, end to end. Runs the vid_out_ada cache
    // collision (§1.3.1) that the per-block stages never touch.
    const out = try forward(&m, a, vid0, txt0, grid, timestep, s);
    defer _ = mlx.mlx_array_free(out);
    try expectStage("out_vid", out, &fx, "out_vid", s);
}


test "seedvr2 dit probe: block 0 attention output" {
    // Localisation probe. `attention` is the only stage with windowing, rope,
    // and the text mean-pool in it, so if the block diverges and this does not,
    // the cause is in the ada/MLP arithmetic instead.
    const a = testing.allocator;
    const s = mlx.gpuStream();
    var fx = try model_mod.loadWeightsSingleFile(a, "src/fixtures/seedvr2_dit_tiny.safetensors");
    defer fx.deinit();
    const cfg = tinyConfig();
    var m = try loadAs(a, &fx, cfg, mlx.mlx_dtype.float32, s);
    defer m.deinit();

    const grid: [3]u32 = .{ 2, 48, 48 };
    const tok_grid: [3]u32 = .{ 2, 24, 24 };

    const vid0 = try fxGet(&fx, "in_vid", s);
    defer _ = mlx.mlx_array_free(vid0);
    const txt0 = try fxGet(&fx, "in_txt", s);
    defer _ = mlx.mlx_array_free(txt0);
    const patched = try patchify(vid0, grid, s);
    defer _ = mlx.mlx_array_free(patched);
    const v = try linT(patched, &m.vid_in_wt, m.vid_in_b, s);
    defer _ = mlx.mlx_array_free(v);
    const t = try linT(txt0, &m.txt_in_wt, m.txt_in_b, s);
    defer _ = mlx.mlx_array_free(t);
    const emb = try embedTimestep(&m, 734.0, s);
    defer _ = mlx.mlx_array_free(emb);
    var mod = try Modulation.init(emb, @intCast(cfg.vid_dim), s);
    defer mod.deinit();

    const b = &m.blocks[0];
    const vn = try rmsNorm(v, null, 1e-5, s);
    defer _ = mlx.mlx_array_free(vn);
    const tn = try rmsNorm(t, null, 1e-5, s);
    defer _ = mlx.mlx_array_free(tn);
    const vm = try adaIn(vn, &mod, LAYER_ATTN, b.vid.adaOf(LAYER_ATTN, KIND_SHIFT), b.vid.adaOf(LAYER_ATTN, KIND_SCALE), s);
    defer _ = mlx.mlx_array_free(vm);
    const ts = b.txtStream();
    const tm = try adaIn(tn, &mod, LAYER_ATTN, ts.adaOf(LAYER_ATTN, KIND_SHIFT), ts.adaOf(LAYER_ATTN, KIND_SCALE), s);
    defer _ = mlx.mlx_array_free(tm);

    var plans = try WindowPlans.build(a, tok_grid);
    defer plans.deinit();
    const at = try attention(b, a, vm, tm, plans.for_(b.method), cfg, s);
    defer _ = mlx.mlx_array_free(at.vid);
    defer _ = mlx.mlx_array_free(at.txt);
    try expectStage("block0_attn_vid", at.vid, &fx, "act_block0_attn_vid", s);
    try expectStage("block0_attn_txt", at.txt, &fx, "act_block0_attn_txt", s);
}

test "seedvr2 dit: the bf16 default reaches the same answer as f32" {
    // THE DEFAULT'S OWN GUARD. The serving path loads bf16 (`computeDtype`)
    // because that is the precision SeedVR2 ships and evaluates at, and
    // because f32 doubles the 3B's residency. That is a claim about ACCURACY,
    // so it is measured here rather than asserted in a comment: the bf16
    // forward is compared against the SAME f32 oracle the staged test uses.
    //
    // The bar is bf16 resolution, not f32's. One bf16 ulp is 2^-8 = 3.9e-3
    // (§1.1.2), the weights themselves are now on that grid, and the error
    // compounds over four blocks — so max relative error is allowed to reach
    // several ulp while the COSINE, which is what catches a structural break,
    // has to stay where the f32 path leaves it.
    const a = testing.allocator;
    const s = mlx.gpuStream();
    var fx = try model_mod.loadWeightsSingleFile(a, "src/fixtures/seedvr2_dit_tiny.safetensors");
    defer fx.deinit();

    const cfg = tinyConfig();
    var m = try loadAs(a, &fx, cfg, mlx.mlx_dtype.bfloat16, s);
    defer m.deinit();

    const vid0 = try fxGet(&fx, "in_vid", s);
    defer _ = mlx.mlx_array_free(vid0);
    const txt0 = try fxGet(&fx, "in_txt", s);
    defer _ = mlx.mlx_array_free(txt0);

    const out = try forward(&m, a, vid0, txt0, .{ 2, 48, 48 }, 734.0, s);
    defer _ = mlx.mlx_array_free(out);
    // The forward hands back f32 whatever it computed in — a caller must not
    // have to know, and the sampler subtracts it from f32 noise.
    try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(out));

    const ref = try fxGet(&fx, "out_vid", s);
    defer _ = mlx.mlx_array_free(ref);
    const c = try cosOf(out, ref, s);
    const e = try maxAbsDiff(out, ref, s);
    const rel = e.diff / (e.scale + 1e-12);
    std.debug.print("[seedvr2-dit] bf16 out_vid: cos={d:.6} max_rel_err={e:.3}\n", .{ c, rel });
    testing.expect(c > 0.999) catch |err| {
        std.debug.print("bf16 forward cosine {d:.6} below 0.999 — this is a STRUCTURAL break, not rounding\n", .{c});
        return err;
    };
    testing.expect(rel < 8.0e-2) catch |err| {
        std.debug.print("bf16 forward max relative error {e:.3} exceeds ~20 bf16 ulp (cos {d:.6})\n", .{ rel, c });
        return err;
    };
}

test "seedvr2 dit: KeyScheme reads the checkpoint's OWN spelling, never the filename" {
    const a = testing.allocator;
    var ours = model_mod.Weights.init(a);
    defer ours.deinit();
    try ours.map.put(try a.dupe(u8, "blocks.0.attn.proj_qkv.txt.weight"), mlx.mlx_array_new_float(0));
    try testing.expectEqual(KeyScheme.ours, KeyScheme.detect(&ours));

    var community = model_mod.Weights.init(a);
    defer community.deinit();
    try community.map.put(try a.dupe(u8, "blocks.0.attn.proj_qkv_txt.weight"), mlx.mlx_array_new_float(0));
    try testing.expectEqual(KeyScheme.mlx_community, KeyScheme.detect(&community));

    // A dir with neither spelling (e.g. only VAE/pos_emb weights probed
    // before the DiT file is even opened) must not silently claim a scheme —
    // `.ours` is the DEFAULT it falls back to, same as every load before
    // this scheme existed.
    var neither = model_mod.Weights.init(a);
    defer neither.deinit();
    try testing.expectEqual(KeyScheme.ours, KeyScheme.detect(&neither));
}

test "seedvr2 dit: a shared layer's attention reads the mirror's per-stream duplicate" {
    const a = testing.allocator;

    // Layers >= mm_layers carry ONE attention weight set serving both streams.
    // Our converter names it `.all`; the mlx-community mirror emits no `_all`
    // attention tensor at all — it materialises that single weight under BOTH
    // stream names instead. Verified on the real int8 pack: across layers
    // 10-31, `proj_qkv`/`proj_out`/`norm_q`/`norm_k` are byte-identical
    // between `_txt` and `_vid`, and no `_all` attn key exists anywhere in the
    // file. So `all` has to resolve to one of the duplicates, or the shared
    // half of the model is unloadable (`MissingSeedVr2DitWeight` on
    // `blocks.10.attn.proj_qkv_all.weight`, live 2026-08-21).
    const shared = try attnPrefix(a, 10, "proj_qkv", "all", .mlx_community);
    defer a.free(shared);
    try testing.expectEqualStrings("blocks.10.attn.proj_qkv_vid", shared);

    // The split layers are untouched: there the two streams are genuinely
    // different tensors and each must keep addressing its own.
    for ([_][]const u8{ "vid", "txt" }) |stream| {
        const split = try attnPrefix(a, 0, "proj_out", stream, .mlx_community);
        defer a.free(split);
        const want = try std.fmt.allocPrint(a, "blocks.0.attn.proj_out_{s}", .{stream});
        defer a.free(want);
        try testing.expectEqualStrings(want, split);
    }

    // Our own converter does emit `.all`, so its spelling must not move.
    const ours = try attnPrefix(a, 10, "norm_q", "all", .ours);
    defer a.free(ours);
    try testing.expectEqualStrings("blocks.10.attn.norm_q.all", ours);
}

test "seedvr2 dit: a quantized DitLinear reaches the same product as the dense weight" {
    // Same template as hunyuan3d.zig's MixedLinear test: quantize a random
    // dense matrix, load it back through `loadMixedLinT`, and check the
    // quantized matmul reproduces the dense product within 8-bit tolerance.
    // This is the primitive the mlx-community 8-bit mirror's DiT linears
    // (qkv/proj_out/mlp/emb_in/txt_in/vid_out) all route through.
    const a = testing.allocator;
    const s = mlx.gpuStream();

    const in: c_int = 128;
    const out: c_int = 64;
    const wv = try a.alloc(f32, @intCast(in * out));
    defer a.free(wv);
    var prng = std.Random.DefaultPrng.init(11);
    for (wv) |*x| x.* = prng.random().float(f32) - 0.5;
    const wsh = [_]c_int{ out, in };
    const wf = mlx.mlx_array_new_data(wv.ptr, &wsh, 2, mlx.mlx_dtype.float32);
    defer _ = mlx.mlx_array_free(wf);

    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    const null_gscale = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_quantize(&triple, wf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", null_gscale, s));
    var qw = mlx.mlx_array_new();
    var qs = mlx.mlx_array_new();
    var qb = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&qw, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&qs, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&qb, triple, 2));

    var ww = model_mod.Weights.init(a);
    defer ww.deinit();
    try ww.map.put(try a.dupe(u8, "l.weight"), qw);
    try ww.map.put(try a.dupe(u8, "l.scales"), qs);
    try ww.map.put(try a.dupe(u8, "l.biases"), qb);
    var dl = try loadMixedLinT(&ww, a, "l", @intCast(in), mlx.mlx_dtype.float32, s);
    defer dl.deinit();
    try testing.expect(dl.quantized);
    try testing.expectEqual(@as(u32, 8), dl.bits);
    try testing.expectEqual(@as(u32, 64), dl.group_size);

    const xv = try a.alloc(f32, @intCast(in));
    defer a.free(xv);
    for (xv, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 7)) * 0.1;
    const xsh = [_]c_int{ 1, in };
    const xa = mlx.mlx_array_new_data(xv.ptr, &xsh, 2, mlx.mlx_dtype.float32);
    defer _ = mlx.mlx_array_free(xa);
    const o = try linT(xa, &dl, null, s);
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_array_eval(o));
    try testing.expectEqual(@as(usize, @intCast(out)), @as(usize, @intCast(mlx.mlx_array_size(o))));
    const od = mlx.mlx_array_data_float32(o) orelse return error.NoData;
    // 8-bit quant of a [-0.5, 0.5] matrix reproduces the dense product closely.
    var manual: f32 = 0;
    for (0..@intCast(in)) |i| manual += xv[i] * wv[i]; // row 0
    try testing.expectApproxEqAbs(manual, od[0], 0.05);

    // Dense weights (no `.scales` sibling) still load through the same
    // chokepoint and reach the plain-matmul branch.
    var dense_w = model_mod.Weights.init(a);
    defer dense_w.deinit();
    const wf2 = mlx.mlx_array_new_data(wv.ptr, &wsh, 2, mlx.mlx_dtype.float32);
    try dense_w.map.put(try a.dupe(u8, "d.weight"), wf2);
    var dd = try loadMixedLinT(&dense_w, a, "d", @intCast(in), mlx.mlx_dtype.float32, s);
    defer dd.deinit();
    try testing.expect(!dd.quantized);
    const od2 = try linT(xa, &dd, null, s);
    defer _ = mlx.mlx_array_free(od2);
    try mlx.check(mlx.mlx_array_eval(od2));
    const od2d = mlx.mlx_array_data_float32(od2) orelse return error.NoData;
    try testing.expectApproxEqAbs(manual, od2d[0], 1e-4);
}

test "the compute dtype is bf16 unless the kill switch is set, and the bill follows it" {
    // The residency estimator multiplies the pack's file sizes by this ratio
    // (`gen.seedvr2PeakBytes`). If the two ever disagree the preflight admits
    // a load it cannot hold — the failure the ratio exists to prevent.
    const dt = computeDtype();
    const ratio = dtypeWidthRatio();
    if (dt == mlx.mlx_dtype.float32) {
        try testing.expectEqual(@as(u64, 2), ratio);
    } else {
        try testing.expectEqual(mlx.mlx_dtype.bfloat16, dt);
        try testing.expectEqual(@as(u64, 1), ratio);
    }
}

test "the window plan is a lossless round trip of the old per-window scatter" {
    // THE OPTIMISATION'S CORRECTNESS, as arithmetic rather than as a picture.
    //
    // `gather` lays every window's rows out end to end and `scatter` takes
    // them back. Composing them must be the identity on the grid: gather row
    // `gather[scatter[r]]` is the row that ends up at `r`, and it has to BE
    // `r` or the reverse mapping silently permutes the video. This is what
    // replaced N full-size `put_along_axis` calls, so it is the one place the
    // whole change could be wrong without any shape disagreeing.
    const a = testing.allocator;
    const grids = [_][3]u32{ .{ 1, 45, 80 }, .{ 2, 24, 24 }, .{ 1, 17, 17 }, .{ 5, 45, 80 }, .{ 1, 30, 53 } };
    for (grids) |grid| {
        for ([_]win_mod.Method{ .plain, .shifted }) |method| {
            var plan = try WindowPlan.build(a, grid, method);
            defer plan.deinit();

            const n: usize = @as(usize, grid[0]) * grid[1] * grid[2];
            try mlx.check(mlx.mlx_array_eval(plan.gather));
            try mlx.check(mlx.mlx_array_eval(plan.scatter));
            const g = mlx.mlx_array_data_int32(plan.gather).?;
            const sc = mlx.mlx_array_data_int32(plan.scatter).?;
            try testing.expectEqual(@as(c_int, @intCast(n)), mlx.getShape(plan.scatter)[0]);

            for (0..n) |r| {
                const slot = sc[r];
                try testing.expect(slot >= 0 and slot < mlx.getShape(plan.gather)[0]);
                testing.expectEqual(@as(i32, @intCast(r)), g[@intCast(slot)]) catch |err| {
                    std.debug.print("grid {any} {s}: row {d} came back from grid row {d}\n", .{
                        grid, @tagName(method), r, g[@intCast(slot)],
                    });
                    return err;
                };
            }

            // Window spans must partition the gathered order exactly, or a
            // window's attention output lands under its neighbour's rows.
            try testing.expectEqual(@as(u32, 0), plan.offsets[0]);
            try testing.expectEqual(@as(usize, plan.windows.len + 1), plan.offsets.len);
            for (plan.windows, 0..) |win, i| {
                try testing.expectEqual(win.tokenCount(), plan.offsets[i + 1] - plan.offsets[i]);
            }
            try testing.expectEqual(
                @as(u32, @intCast(win_mod.totalTokens(plan.windows))),
                plan.offsets[plan.windows.len],
            );

            // Every window points at an extent slot that really holds ITS
            // extent — windows sharing a RoPE table must share a shape.
            for (plan.windows, plan.extent_of) |win, ei| {
                try testing.expectEqual(win.extent(), plan.extents[ei]);
            }
            try testing.expect(plan.extents.len <= plan.windows.len);
        }
    }
}

test "distinct window extents are far fewer than windows on a real grid" {
    // The RoPE tables are built per EXTENT, and that saving is the whole
    // reason the plan carries `extents` at all — so the ratio is the thing
    // worth pinning, not a table of literals that would have to be re-derived
    // every time the window size moves.
    //
    // 720p plain: h is 45 and the window 15, so it divides evenly and every
    // window is 15 tall; w is 80 against 27, so the last column is 26 wide.
    // Nine windows, TWO shapes — the RoPE table count drops 4.5x.
    const a = testing.allocator;
    var plain = try WindowPlan.build(a, .{ 1, 45, 80 }, .plain);
    defer plain.deinit();
    try testing.expectEqual(@as(usize, 9), plain.windows.len);
    try testing.expectEqual(@as(usize, 2), plain.extents.len);

    // The shifted layout has clamped windows at BOTH ends of both axes, so it
    // has more shapes — and still far fewer than it has windows.
    var shifted = try WindowPlan.build(a, .{ 1, 45, 80 }, .shifted);
    defer shifted.deinit();
    try testing.expectEqual(@as(usize, 16), shifted.windows.len);
    try testing.expect(shifted.extents.len * 2 <= shifted.windows.len);

    // A 1080p frame keeps the same window SIZE and grows the COUNT, so the
    // saving grows with resolution rather than washing out at scale.
    var big = try WindowPlan.build(a, .{ 1, 68, 120 }, .plain);
    defer big.deinit();
    try testing.expectEqual(@as(usize, 25), big.windows.len);
    try testing.expect(big.extents.len * 4 <= big.windows.len);
}

test "modulation packing is (d, l, g) with d slowest" {
    // TRAP. A [3, 2, dim] view is the same size and transposed. Build an emb
    // whose value encodes its own flat index and check each slot lands where
    // the reference's `rearrange("b (d l g) -> b d l g")` puts it.
    const s = mlx.gpuStream();
    const dim: c_int = 4;
    const n: usize = @intCast(6 * dim);
    var host: [24]f32 = undefined;
    for (0..n) |i| host[i] = @floatFromInt(i);
    const shp = [_]c_int{ 1, @intCast(n) };
    const emb = mlx.mlx_array_new_data(&host, &shp, shp.len, mlx.mlx_dtype.float32);
    defer _ = mlx.mlx_array_free(emb);

    var mod = try Modulation.init(emb, dim, s);
    defer mod.deinit();

    // Element (d, l, g) sits at d*6 + l*3 + g.
    for ([_]u32{ LAYER_ATTN, LAYER_MLP }) |l| {
        for ([_]u32{ KIND_SHIFT, KIND_SCALE, KIND_GATE }) |k| {
            const got = mod.get(l, k);
            try mlx.check(mlx.mlx_array_eval(got));
            const ptr = mlx.mlx_array_data_float32(got).?;
            for (0..@intCast(dim)) |d| {
                const want: f32 = @floatFromInt(d * 6 + l * 3 + k);
                testing.expectEqual(want, ptr[d]) catch |err| {
                    std.debug.print("slot l={d} k={d} d={d}: want {d} got {d}\n", .{ l, k, d, want, ptr[d] });
                    return err;
                };
            }
        }
    }
}

test "patchify round-trips and orders the patch as (h w c)" {
    const s = mlx.gpuStream();
    const grid: [3]u32 = .{ 1, 4, 4 };
    const c: c_int = 3;
    const n: usize = 1 * 4 * 4 * 3;
    var host: [n]f32 = undefined;
    for (0..n) |i| host[i] = @floatFromInt(i);
    const shp = [_]c_int{ 16, c };
    const x = mlx.mlx_array_new_data(&host, &shp, shp.len, mlx.mlx_dtype.float32);
    defer _ = mlx.mlx_array_free(x);

    const p = try patchify(x, grid, s);
    defer _ = mlx.mlx_array_free(p);
    const pshape = mlx.getShape(p);
    try testing.expectEqual(@as(c_int, 4), pshape[0]); // 1 * 2 * 2 patches
    try testing.expectEqual(@as(c_int, 12), pshape[1]); // 2*2*3

    const back = try unpatchify(p, .{ 1, 2, 2 }, c, s);
    defer _ = mlx.mlx_array_free(back);
    try mlx.check(mlx.mlx_array_eval(back));
    const bp = mlx.mlx_array_data_float32(back).?;
    for (0..n) |i| {
        const want: f32 = @floatFromInt(i);
        testing.expectEqual(want, bp[i]) catch |err| {
            std.debug.print("round-trip mismatch at {d}: want {d} got {d}\n", .{ i, want, bp[i] });
            return err;
        };
    }
}
