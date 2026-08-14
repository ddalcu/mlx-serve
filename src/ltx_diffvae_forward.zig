//! LTX-2.5 DiffVAE decoder — the MLX forward pass.
//!
//! Geometry, the stage ladder and the tiling plan live in `ltx_diffvae.zig`;
//! the fused neighborhood-attention kernel lives in `ltx_diffvae_kernel.zig`.
//! This file is the wiring: load the checkpoint, run the four deterministic
//! stages into a context volume, then denoise patchified pixel tokens against
//! it (v-prediction, 2 Euler steps).
//!
//! Every projection here is dense bf16, so weights are pre-transposed ONCE at
//! load (`[out,in]` → `[in,out]`) and every linear is a plain matmul — a
//! per-call transpose of the 8192x2048 MLP weights is 33 MB of copy per block.
//!
//! Reference: `Lightricks/LTX-2`
//! `ltx_core/model/video_vae/diffusion_video_decoder.py` +
//! `video_vae/transformer/{blocks,layers,swiglu,rope_math,combined/*}.py`.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const ltx = @import("ltx_video.zig");
const geom = @import("ltx_diffvae.zig");
const na = @import("ltx_diffvae_kernel.zig");
const io_util = @import("io_util.zig");
const status = @import("status.zig");

const S = mlx.mlx_stream;

/// Every tensor in the file carries this prefix.
pub const PREFIX = "vae_diffusion_decoder";

/// The file a pack ships the decoder in. Absent → the request is refused by
/// name; it is never a silent downgrade to the conv decoder.
pub const FILE_NAME = "vae_diffusion_decoder.safetensors";

/// Bytes the diffusion stage carries per stage-5 token: context + x + q/k/v at
/// 256 wide plus the f32 RoPE intermediates. MEASURED, not derived — 20.73 GiB
/// peak over 2.38M tokens at 768x512x97 on the shipped checkpoint, billed a
/// little high.
pub const BYTES_PER_STAGE5_TOKEN: u64 = 10 * 1024;

/// Ceiling on one tile's stage-5 tokens. Above this the extra speed flattens
/// and the peak is 20+ GiB; the tile plan cuts until a tile fits.
pub const MAX_TILE_TOKENS: u64 = 3_000_000;

/// Floor, so a starved machine still decodes rather than cutting to nothing.
/// The plan's own per-axis floors bound it further.
pub const MIN_TILE_TOKENS: u64 = 200_000;

/// Share of currently-available system memory one decode may plan against.
const TILE_MEM_SHARE_PCT: u64 = 50;

/// Ceiling the measured decode peak must stay under, in GiB above the resident
/// weights, when the budget is at `MAX_TILE_TOKENS`. Pinned by the peak test.
pub const DECODE_PEAK_BUDGET_GIB: u64 = 24;

/// Tokens per tile from `available` bytes of system memory. The decode transient
/// is a per-REQUEST quantity, so it is sized where it is known rather than billed
/// into the per-MODEL load gate — a gate term would refuse packs that only ever
/// use the conv decoder. `available == 0` means the query failed: take the
/// ceiling rather than refuse to decode.
pub fn tileTokensForMemory(available: u64) u64 {
    if (available == 0) return MAX_TILE_TOKENS;
    const tokens = (available / 100 * TILE_MEM_SHARE_PCT) / BYTES_PER_STAGE5_TOKEN;
    return std.math.clamp(tokens, MIN_TILE_TOKENS, MAX_TILE_TOKENS);
}

var tile_budget_cached: ?u64 = null;

/// The effective per-tile token budget. `MLX_SERVE_DIFFVAE_TILE_TOKENS` pins it
/// (the A/B lever); otherwise it follows free memory.
pub fn tileTokenBudget() u64 {
    if (tile_budget_cached) |v| return v;
    const v: u64 = blk: {
        if (std.c.getenv("MLX_SERVE_DIFFVAE_TILE_TOKENS")) |raw| {
            if (std.fmt.parseInt(u64, std.mem.sliceTo(raw, 0), 10)) |parsed| {
                if (parsed > 0) break :blk parsed;
            } else |_| {}
        }
        break :blk tileTokensForMemory(status.getAvailableMemBytes());
    };
    tile_budget_cached = v;
    return v;
}

/// SwiGLU token tile. Mathematically an identity — at the diffusion stage the
/// 1024-wide hidden over 2.38M tokens is 4.9 GB that never has to exist at once.
const SWIGLU_TILE: c_int = 16384;

// ── small MLX helpers ────────────────────────────────────────────────────

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

fn contig(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&out, x, false, s));
    return out;
}

fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, x, dt, s));
    return out;
}

fn addArr(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, a, b, s));
    return out;
}

fn mulArr(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, a, b, s));
    return out;
}

fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var sg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sg);
    try mlx.check(mlx.mlx_sigmoid(&sg, x, s));
    return mulArr(x, sg, s);
}

/// Half-open slice of one axis of an n-D array (stride 1 on every axis).
fn sliceAxis(x: mlx.mlx_array, axis: usize, lo: c_int, hi: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var start: [8]c_int = @splat(0);
    var stop: [8]c_int = @splat(0);
    var str: [8]c_int = @splat(1);
    for (sh, 0..) |d, i| stop[i] = d;
    start[axis] = lo;
    stop[axis] = hi;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, x, &start, sh.len, &stop, sh.len, &str, sh.len, s));
    return out;
}

/// Grow axis `axis` to `size` by replicating its LAST slice (the reference's
/// `repeat_last`). Already-long-enough is a no-op copy.
fn padRepeatLast(x: mlx.mlx_array, axis: usize, size: c_int, s: S) !mlx.mlx_array {
    const cur = mlx.getShape(x)[axis];
    if (cur >= size) {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&out, x));
        return out;
    }
    const last = try sliceAxis(x, axis, cur - 1, cur, s);
    defer _ = mlx.mlx_array_free(last);
    var rep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rep);
    try mlx.check(mlx.mlx_repeat_axis(&rep, last, size - cur, @intCast(axis), s));
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, x);
    _ = mlx.mlx_vector_array_append_value(vec, rep);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, @intCast(axis), s));
    return out;
}

/// Grow axis `axis` to `size` by edge-replicating BOTH ends (the reference's
/// `symmetric`; leftover goes to the end). Returns the array and how many
/// slices were added at the front, which is what the final crop needs.
fn padSymmetric(x: mlx.mlx_array, axis: usize, size: c_int, s: S) !struct { arr: mlx.mlx_array, before: c_int } {
    const cur = mlx.getShape(x)[axis];
    if (cur >= size) {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&out, x));
        return .{ .arr = out, .before = 0 };
    }
    const need = size - cur;
    const before = @divFloor(need, 2);
    const after = need - before;
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var head: ?mlx.mlx_array = null;
    defer if (head) |h| {
        _ = mlx.mlx_array_free(h);
    };
    if (before > 0) {
        const first = try sliceAxis(x, axis, 0, 1, s);
        defer _ = mlx.mlx_array_free(first);
        var rep = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_repeat_axis(&rep, first, before, @intCast(axis), s));
        head = rep;
        _ = mlx.mlx_vector_array_append_value(vec, rep);
    }
    _ = mlx.mlx_vector_array_append_value(vec, x);
    var tail: ?mlx.mlx_array = null;
    defer if (tail) |t| {
        _ = mlx.mlx_array_free(t);
    };
    if (after > 0) {
        const last = try sliceAxis(x, axis, cur - 1, cur, s);
        defer _ = mlx.mlx_array_free(last);
        var rep = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_repeat_axis(&rep, last, after, @intCast(axis), s));
        tail = rep;
        _ = mlx.mlx_vector_array_append_value(vec, rep);
    }
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, @intCast(axis), s));
    return .{ .arr = out, .before = before };
}

// ── loading ──────────────────────────────────────────────────────────────

/// Load `vae_diffusion_decoder.safetensors` and pre-transpose every 2-D linear
/// weight into `[in, out]`, so the forward never transposes again.
pub fn load(allocator: std.mem.Allocator, path: [:0]const u8, cpu_s: S) !ltx.Component {
    var comp = try ltx.loadComponent(allocator, path, cpu_s);
    errdefer comp.deinit();
    var transposed: u32 = 0;
    var it = comp.map.iterator();
    while (it.next()) |e| {
        const key = e.key_ptr.*;
        const arr = e.value_ptr.*;
        if (!std.mem.endsWith(u8, key, ".weight") or mlx.mlx_array_ndim(arr) != 2) {
            _ = mlx.mlx_array_eval(arr);
            continue;
        }
        const t = try transposeTo(arr, &[_]c_int{ 1, 0 }, cpu_s);
        defer _ = mlx.mlx_array_free(t);
        const c = try contig(t, cpu_s);
        _ = mlx.mlx_array_eval(c);
        _ = mlx.mlx_array_free(arr);
        e.value_ptr.* = c;
        transposed += 1;
    }
    log.info("[diffvae] decoder ready ({d} tensors, {d} pre-transposed)\n", .{ comp.count(), transposed });
    return comp;
}

// ── layer primitives ─────────────────────────────────────────────────────

const Ctx = struct {
    comp: *const ltx.Component,
    cfg: geom.Config,
    alloc: std.mem.Allocator,
    s: S,

    fn weight(self: Ctx, base: []const u8) !mlx.mlx_array {
        var buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}.weight", .{base}) catch return error.DiffVaeKeyTooLong;
        return self.comp.get(key) orelse {
            log.warn("[diffvae] missing weight {s}\n", .{key});
            return error.MissingDiffVaeWeight;
        };
    }

    fn bias(self: Ctx, base: []const u8) ?mlx.mlx_array {
        var buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}.bias", .{base}) catch return null;
        return self.comp.get(key);
    }

    fn tensor(self: Ctx, key: []const u8) !mlx.mlx_array {
        return self.comp.get(key) orelse {
            log.warn("[diffvae] missing tensor {s}\n", .{key});
            return error.MissingDiffVaeWeight;
        };
    }

    /// Dense linear on a channels-last activation; weights are already `[in,out]`.
    fn lin(self: Ctx, x: mlx.mlx_array, base: []const u8) !mlx.mlx_array {
        const w = try self.weight(base);
        var mm = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&mm, x, w, self.s));
        if (self.bias(base)) |b| {
            defer _ = mlx.mlx_array_free(mm);
            return addArr(mm, b, self.s);
        }
        return mm;
    }

    fn rms(self: Ctx, x: mlx.mlx_array, key: []const u8) !mlx.mlx_array {
        const w = try self.tensor(key);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, self.cfg.rms_eps, self.s));
        return out;
    }

    /// `w_down(silu(w_gate x) * w_up x)`, tiled over tokens. The tiling is a
    /// memory measure and mathematically an identity (reference `swiglu_tiled`).
    fn swiglu(self: Ctx, x: mlx.mlx_array, base: []const u8) !mlx.mlx_array {
        var kb: [256]u8 = undefined;
        const gate_k = std.fmt.bufPrint(&kb, "{s}.w_gate", .{base}) catch return error.DiffVaeKeyTooLong;
        const w_gate = try self.weight(gate_k);
        var kb2: [256]u8 = undefined;
        const up_k = std.fmt.bufPrint(&kb2, "{s}.w_up", .{base}) catch return error.DiffVaeKeyTooLong;
        const w_up = try self.weight(up_k);
        var kb3: [256]u8 = undefined;
        const down_k = std.fmt.bufPrint(&kb3, "{s}.w_down", .{base}) catch return error.DiffVaeKeyTooLong;
        const w_down = try self.weight(down_k);

        const sh = mlx.getShape(x);
        const dim = sh[sh.len - 1];
        var n_tok: c_int = 1;
        for (sh[0 .. sh.len - 1]) |d| n_tok *= d;

        const flat = try reshapeTo(x, &[_]c_int{ n_tok, dim }, self.s);
        defer _ = mlx.mlx_array_free(flat);

        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var owned: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (owned.items) |a| _ = mlx.mlx_array_free(a);
            owned.deinit(self.alloc);
        }
        var off: c_int = 0;
        while (off < n_tok) : (off += SWIGLU_TILE) {
            const hi = @min(off + SWIGLU_TILE, n_tok);
            const chunk = if (n_tok <= SWIGLU_TILE) blk: {
                var c = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&c, flat));
                break :blk c;
            } else try sliceAxis(flat, 0, off, hi, self.s);
            defer _ = mlx.mlx_array_free(chunk);
            var g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(g);
            try mlx.check(mlx.mlx_matmul(&g, chunk, w_gate, self.s));
            const act = try silu(g, self.s);
            defer _ = mlx.mlx_array_free(act);
            var u = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(u);
            try mlx.check(mlx.mlx_matmul(&u, chunk, w_up, self.s));
            const prod = try mulArr(act, u, self.s);
            defer _ = mlx.mlx_array_free(prod);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_matmul(&out, prod, w_down, self.s));
            _ = mlx.mlx_array_eval(out);
            try owned.append(self.alloc, out);
            _ = mlx.mlx_vector_array_append_value(vec, out);
        }
        var joined = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(joined);
        if (owned.items.len == 1) {
            try mlx.check(mlx.mlx_array_set(&joined, owned.items[0]));
        } else {
            try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 0, self.s));
        }
        return reshapeTo(joined, sh, self.s);
    }
};

/// `x * (1 + scale) + shift` on a channels-last activation; `scale`/`shift` are
/// `[1, C]` and broadcast over the whole volume.
fn modulate(x: mlx.mlx_array, scale: mlx.mlx_array, shift: mlx.mlx_array, s: S) !mlx.mlx_array {
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var sc1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc1);
    try mlx.check(mlx.mlx_add(&sc1, scale, one, s));
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, x, sc1, s));
    return addArr(scaled, shift, s);
}

// ── absolute per-axis RoPE ───────────────────────────────────────────────

const AxisRope = struct {
    cos: mlx.mlx_array,
    sin: mlx.mlx_array,

    fn deinit(self: *AxisRope) void {
        _ = mlx.mlx_array_free(self.cos);
        _ = mlx.mlx_array_free(self.sin);
    }
};

const Rope = struct {
    axes: [3]AxisRope,
    split: [3]u32,

    fn deinit(self: *Rope) void {
        for (&self.axes) |*a| a.deinit();
    }
};

/// cos/sin for one axis, shaped to broadcast against `[1,T,H,W,NH,D/2,1]`.
/// Positions are LOCAL 0-based: absolute-vs-local RoPE differs by a global
/// phase that cancels inside the softmax over a local window, so every tile
/// rotating from its own origin is identical to rotating from the true one.
fn axisRope(alloc: std.mem.Allocator, axis: usize, length: u32, dim: u32, base: f64, s: S) !AxisRope {
    const half = dim / 2;
    const ang = try alloc.alloc(f32, length * half);
    defer alloc.free(ang);
    for (0..length) |p| {
        for (0..half) |j| {
            ang[p * half + j] = @floatCast(@as(f64, @floatFromInt(p)) * geom.ropeInvFreq(dim, base, @intCast(j)));
        }
    }
    var shape = [_]c_int{ 1, 1, 1, 1, 1, @intCast(half), 1 };
    shape[axis + 1] = @intCast(length);
    const flat_shape = [_]c_int{ @intCast(length), @intCast(half) };
    const a = mlx.mlx_array_new_data(ang.ptr, &flat_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(a);
    const r = try reshapeTo(a, &shape, s);
    defer _ = mlx.mlx_array_free(r);
    var cos = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cos);
    try mlx.check(mlx.mlx_cos(&cos, r, s));
    var sin = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sin(&sin, r, s));
    return .{ .cos = cos, .sin = sin };
}

fn buildRope(alloc: std.mem.Allocator, cfg: geom.Config, dims: [3]u32, s: S) !Rope {
    const split = geom.ropeDimSplit(cfg.head_dim);
    var out: Rope = .{ .axes = undefined, .split = split };
    var built: usize = 0;
    errdefer for (out.axes[0..built]) |*a| a.deinit();
    while (built < 3) : (built += 1) {
        out.axes[built] = try axisRope(alloc, built, dims[built], split[built], cfg.rope_base, s);
    }
    return out;
}

/// Rotate one axis chunk `[1,T,H,W,NH,D]` in f32, returning it in the input
/// dtype. Pairs are ADJACENT (even, odd) within the chunk, not split halves.
fn rotAxis(x: mlx.mlx_array, r: AxisRope, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const dim = sh[5];
    const half = @divExact(dim, 2);
    const pair_shape = [_]c_int{ sh[0], sh[1], sh[2], sh[3], sh[4], half, 2 };
    const in_dtype = mlx.mlx_array_dtype(x);

    const pairs_raw = try reshapeTo(x, &pair_shape, s);
    defer _ = mlx.mlx_array_free(pairs_raw);
    const pairs = try astype(pairs_raw, .float32, s);
    defer _ = mlx.mlx_array_free(pairs);

    const xe = try sliceAxis(pairs, 6, 0, 1, s);
    defer _ = mlx.mlx_array_free(xe);
    const xo = try sliceAxis(pairs, 6, 1, 2, s);
    defer _ = mlx.mlx_array_free(xo);

    const ec = try mulArr(xe, r.cos, s);
    defer _ = mlx.mlx_array_free(ec);
    const os = try mulArr(xo, r.sin, s);
    defer _ = mlx.mlx_array_free(os);
    var re = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(re);
    try mlx.check(mlx.mlx_subtract(&re, ec, os, s));

    const es = try mulArr(xe, r.sin, s);
    defer _ = mlx.mlx_array_free(es);
    const oc = try mulArr(xo, r.cos, s);
    defer _ = mlx.mlx_array_free(oc);
    const ro = try addArr(es, oc, s);
    defer _ = mlx.mlx_array_free(ro);

    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, re);
    _ = mlx.mlx_vector_array_append_value(vec, ro);
    var joined = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(joined);
    try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 6, s));
    const back = try astype(joined, in_dtype, s);
    defer _ = mlx.mlx_array_free(back);
    return reshapeTo(back, sh, s);
}

/// Full (T, H, W) absolute RoPE on `[1,T,H,W,NH,HD]`.
fn applyRope(x: mlx.mlx_array, rope: Rope, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var parts: [3]mlx.mlx_array = undefined;
    var built: usize = 0;
    defer for (parts[0..built]) |p| {
        _ = mlx.mlx_array_free(p);
    };
    var off: c_int = 0;
    while (built < 3) : (built += 1) {
        const d: c_int = @intCast(rope.split[built]);
        const chunk = try sliceAxis(x, 5, off, off + d, s);
        defer _ = mlx.mlx_array_free(chunk);
        parts[built] = try rotAxis(chunk, rope.axes[built], s);
        _ = mlx.mlx_vector_array_append_value(vec, parts[built]);
        off += d;
    }
    std.debug.assert(off == sh[5]);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 5, s));
    return out;
}

// ── attention ────────────────────────────────────────────────────────────

/// `proj(NA(rope(qkv(x))))` — the shared body of the det and diffusion blocks.
/// `x` is the ALREADY normed (and, for the diffusion blocks, modulated) input.
fn attention(c: Ctx, x: mlx.mlx_array, base: []const u8, kern: geom.Kernel, rope: Rope) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const dim = sh[4];
    const hd: c_int = @intCast(c.cfg.head_dim);
    const nh = @divExact(dim, hd);
    const head_shape = [_]c_int{ sh[0], sh[1], sh[2], sh[3], nh, hd };

    var kb: [256]u8 = undefined;
    var qkv: [3]mlx.mlx_array = undefined;
    var built: usize = 0;
    defer for (qkv[0..built]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const names = [3][]const u8{ "to_q", "to_k", "to_v" };
    while (built < 3) : (built += 1) {
        const key = std.fmt.bufPrint(&kb, "{s}.{s}", .{ base, names[built] }) catch return error.DiffVaeKeyTooLong;
        const proj = try c.lin(x, key);
        defer _ = mlx.mlx_array_free(proj);
        qkv[built] = try reshapeTo(proj, &head_shape, c.s);
    }

    // Q/K norm over the head axis, Q pre-scaled by head_dim^-0.5 (the kernel
    // takes scale=1 and expects a scaled Q).
    var qn_key: [256]u8 = undefined;
    const qk = std.fmt.bufPrint(&qn_key, "{s}.q_norm.weight", .{base}) catch return error.DiffVaeKeyTooLong;
    const q_normed = try c.rms(qkv[0], qk);
    defer _ = mlx.mlx_array_free(q_normed);
    var kn_key: [256]u8 = undefined;
    const kk = std.fmt.bufPrint(&kn_key, "{s}.k_norm.weight", .{base}) catch return error.DiffVaeKeyTooLong;
    const k_normed = try c.rms(qkv[1], kk);
    defer _ = mlx.mlx_array_free(k_normed);

    const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(c.cfg.head_dim))));
    defer _ = mlx.mlx_array_free(scale);
    var q_scaled_wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_scaled_wide);
    try mlx.check(mlx.mlx_multiply(&q_scaled_wide, q_normed, scale, c.s));
    // A float scalar promotes a bf16 activation to f32; keep the kernel's three
    // inputs in ONE dtype.
    const q_scaled = try astype(q_scaled_wide, mlx.mlx_array_dtype(qkv[2]), c.s);
    defer _ = mlx.mlx_array_free(q_scaled);

    const q_rope_lazy = try applyRope(q_scaled, rope, c.s);
    defer _ = mlx.mlx_array_free(q_rope_lazy);
    const q_rope = try contig(q_rope_lazy, c.s);
    defer _ = mlx.mlx_array_free(q_rope);
    const k_rope_lazy = try applyRope(k_normed, rope, c.s);
    defer _ = mlx.mlx_array_free(k_rope_lazy);
    const k_rope = try contig(k_rope_lazy, c.s);
    defer _ = mlx.mlx_array_free(k_rope);

    const out = try na.na3d(q_rope, k_rope, qkv[2], kern, c.s);
    defer _ = mlx.mlx_array_free(out);
    var proj_key: [256]u8 = undefined;
    const pk = std.fmt.bufPrint(&proj_key, "{s}.proj", .{base}) catch return error.DiffVaeKeyTooLong;
    return c.lin(out, pk);
}

/// Pre-norm det block: `x + attn(rms(x))` then `x + swiglu(rms(x))`.
fn naBlock(c: Ctx, x: mlx.mlx_array, stage: u32, blk: u32, kern: geom.Kernel, rope: Rope) !mlx.mlx_array {
    var kb: [256]u8 = undefined;
    const n1 = std.fmt.bufPrint(&kb, "{s}.det_stages.{d}.{d}.norm1.weight", .{ PREFIX, stage, blk }) catch return error.DiffVaeKeyTooLong;
    const h1 = try c.rms(x, n1);
    defer _ = mlx.mlx_array_free(h1);
    var ab: [256]u8 = undefined;
    const attn_base = std.fmt.bufPrint(&ab, "{s}.det_stages.{d}.{d}.attn", .{ PREFIX, stage, blk }) catch return error.DiffVaeKeyTooLong;
    const a = try attention(c, h1, attn_base, kern, rope);
    defer _ = mlx.mlx_array_free(a);
    const x1 = try addArr(x, a, c.s);
    defer _ = mlx.mlx_array_free(x1);

    var kb2: [256]u8 = undefined;
    const n2 = std.fmt.bufPrint(&kb2, "{s}.det_stages.{d}.{d}.norm2.weight", .{ PREFIX, stage, blk }) catch return error.DiffVaeKeyTooLong;
    const h2 = try c.rms(x1, n2);
    defer _ = mlx.mlx_array_free(h2);
    var mb: [256]u8 = undefined;
    const mlp_base = std.fmt.bufPrint(&mb, "{s}.det_stages.{d}.{d}.mlp", .{ PREFIX, stage, blk }) catch return error.DiffVaeKeyTooLong;
    const m = try c.swiglu(h2, mlp_base);
    defer _ = mlx.mlx_array_free(m);
    return addArr(x1, m, c.s);
}

/// `proj` then a channels-last 3D pixel shuffle
/// `(b t h w (c p1 p2 p3)) -> (b (t p1) (h p2) (w p3) c)` — channel MINOR in
/// that order, so `c` is the OUTERMOST split of the channel axis. Get it wrong
/// and the picture scrambles at a scale that still looks plausible in motion.
fn upsample(c: Ctx, x: mlx.mlx_array, idx: u32, drop_leading_frame: bool) !mlx.mlx_array {
    var kb: [256]u8 = undefined;
    const base = std.fmt.bufPrint(&kb, "{s}.upsamples.{d}.proj", .{ PREFIX, idx }) catch return error.DiffVaeKeyTooLong;
    const p = try c.lin(x, base);
    defer _ = mlx.mlx_array_free(p);

    const up = c.cfg.upsamples[idx];
    const sh = mlx.getShape(p);
    const p1: c_int = @intCast(up.stride[0]);
    const p2: c_int = @intCast(up.stride[1]);
    const p3: c_int = @intCast(up.stride[2]);
    const ch = @divExact(sh[4], p1 * p2 * p3);
    const split = try reshapeTo(p, &[_]c_int{ sh[0], sh[1], sh[2], sh[3], ch, p1, p2, p3 }, c.s);
    defer _ = mlx.mlx_array_free(split);
    const t = try transposeTo(split, &[_]c_int{ 0, 1, 5, 2, 6, 3, 7, 4 }, c.s);
    defer _ = mlx.mlx_array_free(t);
    const merged = try reshapeTo(t, &[_]c_int{ sh[0], sh[1] * p1, sh[2] * p2, sh[3] * p3, ch }, c.s);
    defer _ = mlx.mlx_array_free(merged);

    if (up.stride[0] == 2 and drop_leading_frame) {
        const frames = mlx.getShape(merged)[1];
        const dropped = try sliceAxis(merged, 1, 1, frames, c.s);
        defer _ = mlx.mlx_array_free(dropped);
        return contig(dropped, c.s);
    }
    return contig(merged, c.s);
}

/// One deterministic stage: its NA blocks, then its upsample.
fn detStage(c: Ctx, x_in: mlx.mlx_array, stage: u32, drop_leading_frame: bool) !mlx.mlx_array {
    const spec = c.cfg.stages[stage];
    const sh = mlx.getShape(x_in);
    var rope = try buildRope(c.alloc, c.cfg, .{ @intCast(sh[1]), @intCast(sh[2]), @intCast(sh[3]) }, c.s);
    defer rope.deinit();

    var x = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&x, x_in));
    // ONE owner: `defer` frees whatever `x` holds at scope exit, on the error
    // path too. An `errdefer` here plus a `defer` later is the same handle
    // freed twice when a later step fails.
    defer _ = mlx.mlx_array_free(x);
    for (0..spec.depth) |b| {
        const nx = try naBlock(c, x, stage, @intCast(b), spec.kernel, rope);
        _ = mlx.mlx_array_free(x);
        x = nx;
        _ = mlx.mlx_array_eval(x);
    }
    return upsample(c, x, stage, drop_leading_frame);
}

// ── the diffusion stage ──────────────────────────────────────────────────

/// `[1, 7*dim]` = `shared_adaln.proj(silu(t_embedder(t)))`. Chunk order is
/// scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp, gate_ctx;
/// the gates are unused in this pathway (folded into the linears at export).
fn modulation(c: Ctx, t: f32) !mlx.mlx_array {
    const sinus = try ltx.timestepSinusoid(t, 256, c.s);
    defer _ = mlx.mlx_array_free(sinus);
    var kb: [256]u8 = undefined;
    const l1 = std.fmt.bufPrint(&kb, "{s}.t_embedder.timestep_embedder.linear1", .{PREFIX}) catch return error.DiffVaeKeyTooLong;
    const w = try c.weight(l1);
    const sin_typed = try astype(sinus, mlx.mlx_array_dtype(w), c.s);
    defer _ = mlx.mlx_array_free(sin_typed);
    const e1 = try c.lin(sin_typed, l1);
    defer _ = mlx.mlx_array_free(e1);
    const a1 = try silu(e1, c.s);
    defer _ = mlx.mlx_array_free(a1);
    var kb2: [256]u8 = undefined;
    const l2 = std.fmt.bufPrint(&kb2, "{s}.t_embedder.timestep_embedder.linear2", .{PREFIX}) catch return error.DiffVaeKeyTooLong;
    const emb = try c.lin(a1, l2);
    defer _ = mlx.mlx_array_free(emb);
    const act = try silu(emb, c.s);
    defer _ = mlx.mlx_array_free(act);
    var kb3: [256]u8 = undefined;
    const ad = std.fmt.bufPrint(&kb3, "{s}.shared_adaln.proj", .{PREFIX}) catch return error.DiffVaeKeyTooLong;
    return c.lin(act, ad);
}

/// Chunk `i` of the modulation plus this block's own `scale_shift_table` row.
fn modChunk(c: Ctx, mod: mlx.mlx_array, table: mlx.mlx_array, i: c_int) !mlx.mlx_array {
    const dim: c_int = @intCast(c.cfg.stage5_dim);
    const m = try sliceAxis(mod, 1, i * dim, (i + 1) * dim, c.s);
    defer _ = mlx.mlx_array_free(m);
    const row = try sliceAxis(table, 0, i, i + 1, c.s);
    defer _ = mlx.mlx_array_free(row);
    return addArr(m, row, c.s);
}

/// One `CombinedDiffusionNABlock`: inject the context, then an AdaLN-modulated
/// NA residual and an AdaLN-modulated SwiGLU residual.
fn diffBlock(c: Ctx, x: mlx.mlx_array, ctx_vol: mlx.mlx_array, blk: u32, mod: mlx.mlx_array, rope: Rope) !mlx.mlx_array {
    var kb: [256]u8 = undefined;
    const tk = std.fmt.bufPrint(&kb, "{s}.diff_blocks.{d}.scale_shift_table", .{ PREFIX, blk }) catch return error.DiffVaeKeyTooLong;
    const table = try c.tensor(tk);
    const scale_msa = try modChunk(c, mod, table, 0);
    defer _ = mlx.mlx_array_free(scale_msa);
    const shift_msa = try modChunk(c, mod, table, 1);
    defer _ = mlx.mlx_array_free(shift_msa);
    const scale_mlp = try modChunk(c, mod, table, 3);
    defer _ = mlx.mlx_array_free(scale_mlp);
    const shift_mlp = try modChunk(c, mod, table, 4);
    defer _ = mlx.mlx_array_free(shift_mlp);

    var cb: [256]u8 = undefined;
    const ck = std.fmt.bufPrint(&cb, "{s}.diff_blocks.{d}.context_proj", .{ PREFIX, blk }) catch return error.DiffVaeKeyTooLong;
    const ctx_proj = try c.lin(ctx_vol, ck);
    defer _ = mlx.mlx_array_free(ctx_proj);
    var h = try addArr(x, ctx_proj, c.s);
    errdefer _ = mlx.mlx_array_free(h);

    {
        var nb: [256]u8 = undefined;
        const n1 = std.fmt.bufPrint(&nb, "{s}.diff_blocks.{d}.norm1.weight", .{ PREFIX, blk }) catch return error.DiffVaeKeyTooLong;
        const normed = try c.rms(h, n1);
        defer _ = mlx.mlx_array_free(normed);
        const modded = try modulate(normed, scale_msa, shift_msa, c.s);
        defer _ = mlx.mlx_array_free(modded);
        const modded_t = try astype(modded, mlx.mlx_array_dtype(h), c.s);
        defer _ = mlx.mlx_array_free(modded_t);
        var ab: [256]u8 = undefined;
        const attn_base = std.fmt.bufPrint(&ab, "{s}.diff_blocks.{d}.attn", .{ PREFIX, blk }) catch return error.DiffVaeKeyTooLong;
        const a = try attention(c, modded_t, attn_base, c.cfg.stage5_kernel, rope);
        defer _ = mlx.mlx_array_free(a);
        const nh = try addArr(h, a, c.s);
        _ = mlx.mlx_array_free(h);
        h = nh;
    }
    {
        var nb: [256]u8 = undefined;
        const n2 = std.fmt.bufPrint(&nb, "{s}.diff_blocks.{d}.norm2.weight", .{ PREFIX, blk }) catch return error.DiffVaeKeyTooLong;
        const normed = try c.rms(h, n2);
        defer _ = mlx.mlx_array_free(normed);
        const modded = try modulate(normed, scale_mlp, shift_mlp, c.s);
        defer _ = mlx.mlx_array_free(modded);
        const modded_t = try astype(modded, mlx.mlx_array_dtype(h), c.s);
        defer _ = mlx.mlx_array_free(modded_t);
        var mb: [256]u8 = undefined;
        const mlp_base = std.fmt.bufPrint(&mb, "{s}.diff_blocks.{d}.mlp", .{ PREFIX, blk }) catch return error.DiffVaeKeyTooLong;
        const m = try c.swiglu(modded_t, mlp_base);
        defer _ = mlx.mlx_array_free(m);
        const nh = try addArr(h, m, c.s);
        _ = mlx.mlx_array_free(h);
        h = nh;
    }
    return h;
}

/// One diffusion step: the model's prediction for `x_t`, in PATCHIFIED pixel
/// space `[1, F, H, W, 48]` (the unpatchify is done once, per tile, at the end).
fn diffStep(c: Ctx, ctx_vol: mlx.mlx_array, x_t: mlx.mlx_array, t: f32, rope: Rope) !mlx.mlx_array {
    const mod = try modulation(c, t);
    defer _ = mlx.mlx_array_free(mod);
    var kb: [256]u8 = undefined;
    const ck = std.fmt.bufPrint(&kb, "{s}.conv_in_x_t", .{PREFIX}) catch return error.DiffVaeKeyTooLong;
    var x = try c.lin(x_t, ck);
    defer _ = mlx.mlx_array_free(x);
    for (0..c.cfg.stage5_depth) |b| {
        const nx = try diffBlock(c, x, ctx_vol, @intCast(b), mod, rope);
        _ = mlx.mlx_array_free(x);
        x = nx;
        _ = mlx.mlx_array_eval(x);
    }
    var nb: [256]u8 = undefined;
    const nk = std.fmt.bufPrint(&nb, "{s}.norm_out.weight", .{PREFIX}) catch return error.DiffVaeKeyTooLong;
    const normed = try c.rms(x, nk);
    defer _ = mlx.mlx_array_free(normed);
    var ob: [256]u8 = undefined;
    const ok = std.fmt.bufPrint(&ob, "{s}.conv_out", .{PREFIX}) catch return error.DiffVaeKeyTooLong;
    return c.lin(normed, ok);
}

/// `x - (t_now - t_next) * v`, in f32 (the reference upcasts both sides).
fn eulerStep(x_t: mlx.mlx_array, v: mlx.mlx_array, t_now: f32, t_next: f32, s: S) !mlx.mlx_array {
    const dt = mlx.mlx_array_new_float(geom.eulerStepScale(t_now, t_next));
    defer _ = mlx.mlx_array_free(dt);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, v, dt, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&out, x_t, scaled, s));
    return out;
}

/// Channels-last unpatchify `[B,F,H,W, C*p*p] -> [B,F, H*p, W*p, C]`, channel
/// split `(c, r=W, q=H)` — width before height, matching `ops.unpatchify`.
fn unpatchify(x: mlx.mlx_array, ps: u32, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const p: c_int = @intCast(ps);
    const ch = @divExact(sh[4], p * p);
    const split = try reshapeTo(x, &[_]c_int{ sh[0], sh[1], sh[2], sh[3], ch, p, p }, s);
    defer _ = mlx.mlx_array_free(split);
    const t = try transposeTo(split, &[_]c_int{ 0, 1, 2, 6, 3, 5, 4 }, s);
    defer _ = mlx.mlx_array_free(t);
    const merged = try reshapeTo(t, &[_]c_int{ sh[0], sh[1], sh[2] * p, sh[3] * p, ch }, s);
    defer _ = mlx.mlx_array_free(merged);
    return contig(merged, s);
}

// ── tiling plan ──────────────────────────────────────────────────────────

pub const TilePlan = struct {
    /// Max tile extent per axis, on the stage-4 input grid.
    tile: [3]u32,
    /// Seam overlap per axis, stage-4 units (= the larger of the two halos).
    overlap: [3]u32,
};

/// Cut the stage-4 grid until one tile's diffusion-stage token count fits the
/// budget, always halving the axis that is currently the most oversized. Stages
/// 1-3 are NOT tiled — they run on the full volume, which is 1.6M tokens at dim
/// 512 even for a 1920x1088 clip.
pub fn planTiles(cfg: geom.Config, s4: [3]u32, budget_tokens: u64) TilePlan {
    const halos = geom.tileHalos(cfg);
    const mins = geom.tileMinSize(cfg);
    var overlap: [3]u32 = undefined;
    var floor: [3]u32 = undefined;
    for (0..3) |a| {
        overlap[a] = @max(halos[0][a], halos[1][a]);
        floor[a] = @max(mins[a], 2 * overlap[a]);
    }
    const up = cfg.upsamples[3].stride;
    var tile = s4;
    var guard: u32 = 0;
    while (guard < 64) : (guard += 1) {
        const m = geom.AxisMap{ .scale = up[0], .temporal = true };
        const tokens: u64 = @as(u64, m.map(tile[0])) * (tile[1] * up[1]) * (tile[2] * up[2]);
        if (tokens <= budget_tokens) break;
        // Halve whichever axis has the most room left above its own floor.
        var pick: usize = 0;
        var best: f64 = 0;
        for (0..3) |a| {
            if (tile[a] <= floor[a]) continue;
            const r = @as(f64, @floatFromInt(tile[a])) / @as(f64, @floatFromInt(floor[a]));
            if (r > best) {
                best = r;
                pick = a;
            }
        }
        if (best == 0) break; // every axis is at its floor
        tile[pick] = @max(floor[pick], (tile[pick] + 1) / 2);
    }
    return .{ .tile = tile, .overlap = overlap };
}

// ── decode ───────────────────────────────────────────────────────────────

const MAX_TILES_PER_AXIS = 24;

/// What the diffusion stage PREDICTS, how many steps it takes, and what the
/// timestep is scaled by before the sinusoid.
///
/// These are constructor arguments in the reference, read from a `vae` config —
/// and NO pack ships one, so the class defaults (v-prediction, 2 steps, x1) are
/// a guess, not a contract. Lightricks call this file a "DiffVAE 1-step x0
/// decoder", and that is what the weights actually behave as: run as 2-step
/// v-prediction the model's output is small against the noise it is subtracted
/// from and the decode comes back as static (measured adjacent-pixel gradient
/// 44 on a 0-255 frame, against 2.8 for the conv decoder — the `lora_noise`
/// static signature). x0 at one step is the arm that produces a picture.
///
/// Kept as a struct with kill switches because none of it is declared anywhere:
/// a pack that ships a real config, or a future decoder, changes these.
pub const Sampler = struct {
    /// `true` → the model predicts x0 and the last step's output IS the result.
    /// `false` → it predicts a velocity and every step is a reverse Euler update.
    predict_x0: bool = true,
    num_steps: u32 = 1,
    /// Multiplies the timestep before `t_embedder`. LTX scales timesteps by
    /// 1000 everywhere else in the pack (`config.json`
    /// `timestep_scale_multiplier`), and the reference's own default of 1.0 is
    /// the class default rather than this checkpoint's.
    timestep_scale: f32 = 1000.0,

    pub const default = Sampler{};

    /// `MLX_SERVE_DIFFVAE_{OUTPUT,STEPS,TSCALE}` override it, one field each.
    pub fn resolved() Sampler {
        var out = Sampler.default;
        if (std.c.getenv("MLX_SERVE_DIFFVAE_OUTPUT")) |raw| {
            out.predict_x0 = std.mem.eql(u8, std.mem.sliceTo(raw, 0), "x0");
        }
        if (std.c.getenv("MLX_SERVE_DIFFVAE_STEPS")) |raw| {
            if (std.fmt.parseInt(u32, std.mem.sliceTo(raw, 0), 10)) |v| {
                if (v > 0) out.num_steps = v;
            } else |_| {}
        }
        if (std.c.getenv("MLX_SERVE_DIFFVAE_TSCALE")) |raw| {
            out.timestep_scale = std.fmt.parseFloat(f32, std.mem.sliceTo(raw, 0)) catch out.timestep_scale;
        }
        return out;
    }
};

pub const Options = struct {
    seed: u64 = 0,
    num_steps: u32 = 0, // 0 → `Sampler.resolved()`
    /// 0 → `tileTokenBudget()`.
    tile_tokens: u64 = 0,
    sampler: Sampler = Sampler.default,
};

/// Decode an LTX latent `[1,128,T,H,W]` (BCFHW) to pixels `[1,3,F,H*32,W*32]`
/// in `[-1,1]` — the same contract as `ltx_video.vaeDecode`, so the two are
/// interchangeable at the call site.
pub fn decode(
    allocator: std.mem.Allocator,
    comp: *const ltx.Component,
    cfg: geom.Config,
    latent_bcfhw: mlx.mlx_array,
    opts: Options,
    s: S,
) !mlx.mlx_array {
    var c_opts = opts;
    c_opts.sampler = Sampler.resolved();
    const c = Ctx{ .comp = comp, .cfg = cfg, .alloc = allocator, .s = s };
    const lsh = mlx.getShape(latent_bcfhw);
    if (lsh.len != 5) return error.DiffVaeBadLatentRank;
    const content_latent = [3]u32{ @intCast(lsh[2]), @intCast(lsh[3]), @intCast(lsh[4]) };
    const content_px = geom.pixelShape(cfg, content_latent);

    // BCFHW → BFHWC, then pad up to the latent floor every stage's NA needs.
    var lat = try transposeTo(latent_bcfhw, &[_]c_int{ 0, 2, 3, 4, 1 }, s);
    defer _ = mlx.mlx_array_free(lat);
    {
        const cont = try contig(lat, s);
        _ = mlx.mlx_array_free(lat);
        lat = cont;
    }
    const floor = geom.allStagesMinTile(cfg);
    var h_pad_before: u32 = 0;
    var w_pad_before: u32 = 0;
    {
        const t_padded = try padRepeatLast(lat, 1, @intCast(floor[0]), s);
        _ = mlx.mlx_array_free(lat);
        lat = t_padded;
        const hp = try padSymmetric(lat, 2, @intCast(floor[1]), s);
        _ = mlx.mlx_array_free(lat);
        lat = hp.arr;
        h_pad_before = @intCast(hp.before);
        const wp = try padSymmetric(lat, 3, @intCast(floor[2]), s);
        _ = mlx.mlx_array_free(lat);
        lat = wp.arr;
        w_pad_before = @intCast(wp.before);
    }
    const work_latent = [3]u32{
        @intCast(mlx.getShape(lat)[1]),
        @intCast(mlx.getShape(lat)[2]),
        @intCast(mlx.getShape(lat)[3]),
    };
    const work_px = geom.pixelShape(cfg, work_latent);

    // NATTEN's window shifts inward at the LAST frame, so the trailing latent
    // frame is replicated through stages 1-4 and the appendix cropped off the
    // context again before the diffusion stage.
    const ghost = cfg.trailingPadLatentFrames();
    {
        const padded = try padRepeatLast(lat, 1, @intCast(work_latent[0] + ghost), s);
        _ = mlx.mlx_array_free(lat);
        lat = padded;
    }

    // un-normalize → conv_in → stages 1-3, on the FULL volume.
    var x: mlx.mlx_array = blk: {
        var mean_key: [128]u8 = undefined;
        const mk = try std.fmt.bufPrint(&mean_key, "{s}.per_channel_statistics.mean", .{PREFIX});
        var std_key: [128]u8 = undefined;
        const sk = try std.fmt.bufPrint(&std_key, "{s}.per_channel_statistics.std", .{PREFIX});
        const mean = try c.tensor(mk);
        const stdv = try c.tensor(sk);
        const scaled = try mulArr(lat, stdv, s);
        defer _ = mlx.mlx_array_free(scaled);
        const shifted = try addArr(scaled, mean, s);
        defer _ = mlx.mlx_array_free(shifted);
        var ck: [128]u8 = undefined;
        const conv_in = try std.fmt.bufPrint(&ck, "{s}.conv_in", .{PREFIX});
        const w = try c.weight(conv_in);
        const typed = try astype(shifted, mlx.mlx_array_dtype(w), s);
        defer _ = mlx.mlx_array_free(typed);
        break :blk try c.lin(typed, conv_in);
    };
    defer _ = mlx.mlx_array_free(x);
    for (0..3) |i| {
        const nx = try detStage(c, x, @intCast(i), true);
        _ = mlx.mlx_array_free(x);
        x = nx;
        _ = mlx.mlx_array_eval(x);
        _ = mlx.mlx_clear_cache();
    }
    const feat_s4 = x; // alias — `x`'s defer above is its one owner

    // Content extent on the stage-4 grid (the ghost frames sit past it).
    const s4 = geom.stage4FromLatent(cfg, work_latent, true);
    const s4_total: u32 = @intCast(mlx.getShape(feat_s4)[1]);
    const scale = geom.stage4PixelScale(cfg);

    // Noise for the WHOLE target, sliced per tile — independent per-tile noise
    // is a seam generator. Patchified layout: the diffusion stage never leaves it.
    const patch: u32 = cfg.patch_size;
    const noise_shape = [_]c_int{
        1,
        @intCast(work_px[0]),
        @intCast(work_px[1] / patch),
        @intCast(work_px[2] / patch),
        @intCast(cfg.patchChannels()),
    };
    const noise = try randomNormal(&noise_shape, opts.seed, s);
    defer _ = mlx.mlx_array_free(noise);

    const plan = planTiles(cfg, s4, if (opts.tile_tokens > 0) opts.tile_tokens else tileTokenBudget());
    var tbuf: [MAX_TILES_PER_AXIS]geom.Interval = undefined;
    var hbuf: [MAX_TILES_PER_AXIS]geom.Interval = undefined;
    var wbuf: [MAX_TILES_PER_AXIS]geom.Interval = undefined;
    const t_tiles = geom.splitAxis(s4[0], plan.tile[0], plan.overlap[0], geom.tileMinSize(cfg)[0], .{ .scale = scale[0], .temporal = true }, &tbuf);
    const h_tiles = geom.splitAxis(s4[1], plan.tile[1], plan.overlap[1], geom.tileMinSize(cfg)[1], .{ .scale = scale[1] }, &hbuf);
    const w_tiles = geom.splitAxis(s4[2], plan.tile[2], plan.overlap[2], geom.tileMinSize(cfg)[2], .{ .scale = scale[2] }, &wbuf);
    const n_tiles = t_tiles.len * h_tiles.len * w_tiles.len;
    log.info(
        "[diffvae] decoding {d}x{d}x{d} px in {d} tile(s) (stage-4 grid {d}x{d}x{d}, tile {d}x{d}x{d})\n",
        .{ content_px[0], content_px[1], content_px[2], n_tiles, s4[0], s4[1], s4[2], plan.tile[0], plan.tile[1], plan.tile[2] },
    );

    var accum: ?mlx.mlx_array = null;
    errdefer if (accum) |a| {
        _ = mlx.mlx_array_free(a);
    };

    for (t_tiles) |tt| {
        for (h_tiles) |ht| {
            for (w_tiles) |wt| {
                const is_origin = tt.start == 0;
                const pad_trailing = tt.end == s4[0];
                const t_hi: c_int = if (pad_trailing) @intCast(s4_total) else @intCast(tt.end);
                const feat = try sliceTile(feat_s4, .{ @intCast(tt.start), @intCast(ht.start), @intCast(wt.start) }, .{ t_hi, @intCast(ht.end), @intCast(wt.end) }, s);
                defer _ = mlx.mlx_array_free(feat);

                const tile_px = try decodeTile(c, feat, noise, tt, ht, wt, is_origin, pad_trailing, c_opts);
                defer _ = mlx.mlx_array_free(tile_px);

                if (n_tiles == 1) {
                    accum = try astype(tile_px, .float32, s);
                } else {
                    const weighted = try applyTileWeights(allocator, tile_px, tt, ht, wt, s);
                    defer _ = mlx.mlx_array_free(weighted);
                    accum = try blendInto(accum, weighted, tt, ht, wt, work_px, cfg, s);
                }
                _ = mlx.mlx_clear_cache();
            }
        }
    }

    var pixels = accum.?;
    accum = null;
    defer _ = mlx.mlx_array_free(pixels);

    // Crop the size-floor pad back off, then BFHWC → BCFHW.
    if (!std.meta.eql(work_px, content_px)) {
        const sp = geom.latentSpatialScale(cfg);
        const cropped = try cropContent(pixels, content_px, h_pad_before * sp[0], w_pad_before * sp[1], s);
        _ = mlx.mlx_array_free(pixels);
        pixels = cropped;
    }
    const t = try transposeTo(pixels, &[_]c_int{ 0, 4, 1, 2, 3 }, s);
    defer _ = mlx.mlx_array_free(t);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&out, t, false, s));
    _ = mlx.mlx_array_eval(out);
    _ = mlx.mlx_clear_cache();
    return out;
}

fn randomNormal(shape: []const c_int, seed: u64, s: S) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&out, shape.ptr, shape.len, .float32, 0.0, 1.0, key, s));
    return out;
}

fn sliceTile(x: mlx.mlx_array, lo: [3]c_int, hi: [3]c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const start = [_]c_int{ 0, lo[0], lo[1], lo[2], 0 };
    const stop = [_]c_int{ sh[0], hi[0], hi[1], hi[2], sh[4] };
    const str = [_]c_int{ 1, 1, 1, 1, 1 };
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    try mlx.check(mlx.mlx_slice(&sliced, x, &start, 5, &stop, 5, &str, 5, s));
    return contig(sliced, s);
}

/// Stage 4 + the diffusion loop on one stage-4 feature tile, returning pixels
/// `[1, F, H, W, 3]` for that tile's own extent.
fn decodeTile(
    c: Ctx,
    feat: mlx.mlx_array,
    noise: mlx.mlx_array,
    tt: geom.Interval,
    ht: geom.Interval,
    wt: geom.Interval,
    is_origin: bool,
    pad_trailing: bool,
    opts: Options,
) !mlx.mlx_array {
    const s = c.s;
    var ctx_vol = try detStage(c, feat, 3, is_origin);
    defer _ = mlx.mlx_array_free(ctx_vol);
    if (pad_trailing) {
        const keep = geom.contextKeepFrames(c.cfg, @intCast(mlx.getShape(ctx_vol)[1]), geom.latentTimeScale(c.cfg));
        const cropped = try sliceAxis(ctx_vol, 1, 0, @intCast(keep), s);
        defer _ = mlx.mlx_array_free(cropped);
        // Materialize BEFORE releasing the old handle, so a failure here leaves
        // `ctx_vol` valid for the defer rather than dangling.
        const packed_ctx = try contig(cropped, s);
        _ = mlx.mlx_array_free(ctx_vol);
        ctx_vol = packed_ctx;
    }
    _ = mlx.mlx_array_eval(ctx_vol);
    _ = mlx.mlx_clear_cache();

    const csh = mlx.getShape(ctx_vol);
    const frames = csh[1];
    const patch: c_int = @intCast(c.cfg.patch_size);

    // x_t: this tile's slice of the ONE global noise field, grown to the
    // context's own canvas with the same edge policy as the size floor (NA
    // mixes padded values into kept pixels near the boundary; fresh noise there
    // would be a different distribution).
    var x_t = blk: {
        const nsh = mlx.getShape(noise);
        const start = [_]c_int{ 0, @intCast(tt.out_start), @intCast(ht.out_start / @as(u32, @intCast(patch))), @intCast(wt.out_start / @as(u32, @intCast(patch))), 0 };
        const stop = [_]c_int{ nsh[0], @intCast(tt.out_end), @intCast(ht.out_end / @as(u32, @intCast(patch))), @intCast(wt.out_end / @as(u32, @intCast(patch))), nsh[4] };
        const str = [_]c_int{ 1, 1, 1, 1, 1 };
        var sl = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sl);
        try mlx.check(mlx.mlx_slice(&sl, noise, &start, 5, &stop, 5, &str, 5, s));
        var grown = try padRepeatLast(sl, 1, frames, s);
        if (mlx.getShape(grown)[2] != csh[2]) {
            const r = try padSymmetric(grown, 2, csh[2], s);
            _ = mlx.mlx_array_free(grown);
            grown = r.arr;
        }
        if (mlx.getShape(grown)[3] != csh[3]) {
            const r = try padSymmetric(grown, 3, csh[3], s);
            _ = mlx.mlx_array_free(grown);
            grown = r.arr;
        }
        const cg = try contig(grown, s);
        _ = mlx.mlx_array_free(grown);
        break :blk cg;
    };
    defer _ = mlx.mlx_array_free(x_t);

    var rope = try buildRope(c.alloc, c.cfg, .{ @intCast(csh[1]), @intCast(csh[2]), @intCast(csh[3]) }, s);
    defer rope.deinit();

    const sampler = opts.sampler;
    const n_steps = if (opts.num_steps > 0) opts.num_steps else sampler.num_steps;
    var tbuf: [16]f32 = undefined;
    const ts = geom.timesteps(@min(n_steps, tbuf.len), &tbuf);
    const model_dtype = mlx.mlx_array_dtype(ctx_vol);

    // Every step but the last is a reverse Euler update; the LAST one either IS
    // the answer (x0) or is Euler'd down to t=0 (v). Mirrors the reference's
    // `_decode_one_tile`, whose x0 arm returns `model_out` untouched.
    for (ts, 0..) |t_now, i| {
        const last = i + 1 == ts.len;
        const x_in = try astype(x_t, model_dtype, s);
        defer _ = mlx.mlx_array_free(x_in);
        const pred_raw = try diffStep(c, ctx_vol, x_in, t_now * sampler.timestep_scale, rope);
        defer _ = mlx.mlx_array_free(pred_raw);
        const pred = try astype(pred_raw, .float32, s);
        defer _ = mlx.mlx_array_free(pred);
        if (last and sampler.predict_x0) {
            _ = mlx.mlx_array_free(x_t);
            x_t = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&x_t, pred));
        } else {
            const t_next: f32 = if (last) 0.0 else ts[i + 1];
            const nx = try eulerStep(x_t, pred, t_now, t_next, s);
            _ = mlx.mlx_array_free(x_t);
            x_t = nx;
        }
        _ = mlx.mlx_array_eval(x_t);
        _ = mlx.mlx_clear_cache();
    }
    return unpatchify(x_t, c.cfg.patch_size, s);
}

/// Crop a tile's pixels back to the extent it OWNS (a trailing tile decodes
/// extra ghost frames) and scale by the separable trapezoid blend weights.
fn applyTileWeights(
    alloc: std.mem.Allocator,
    tile_px: mlx.mlx_array,
    tt: geom.Interval,
    ht: geom.Interval,
    wt: geom.Interval,
    s: S,
) !mlx.mlx_array {
    var x = try astype(tile_px, .float32, s);
    errdefer _ = mlx.mlx_array_free(x);
    const want = [3]c_int{ @intCast(tt.outLen()), @intCast(ht.outLen()), @intCast(wt.outLen()) };
    for (0..3) |a| {
        if (mlx.getShape(x)[a + 1] != want[a]) {
            const cropped = try sliceAxis(x, a + 1, 0, want[a], s);
            _ = mlx.mlx_array_free(x);
            x = try contig(cropped, s);
            _ = mlx.mlx_array_free(cropped);
        }
    }
    const ivs = [3]geom.Interval{ tt, ht, wt };
    for (ivs, 0..) |iv, a| {
        if (iv.left_ramp == 0 and iv.right_ramp == 0) continue;
        const n = iv.outLen();
        const w = try alloc.alloc(f32, n);
        defer alloc.free(w);
        for (0..n) |i| w[i] = geom.tileWeight(iv, @intCast(i));
        var shape = [_]c_int{ 1, 1, 1, 1, 1 };
        shape[a + 1] = @intCast(n);
        const wa = mlx.mlx_array_new_data(w.ptr, &shape, 5, .float32);
        defer _ = mlx.mlx_array_free(wa);
        const scaled = try mulArr(x, wa, s);
        _ = mlx.mlx_array_free(x);
        x = scaled;
    }
    return x;
}

/// Add a weighted tile into the full-frame accumulator at its own coordinates.
fn blendInto(
    accum: ?mlx.mlx_array,
    tile: mlx.mlx_array,
    tt: geom.Interval,
    ht: geom.Interval,
    wt: geom.Interval,
    work_px: [3]u32,
    cfg: geom.Config,
    s: S,
) !mlx.mlx_array {
    const buf = accum orelse blk: {
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);
        const shape = [_]c_int{ 1, @intCast(work_px[0]), @intCast(work_px[1]), @intCast(work_px[2]), @intCast(cfg.out_channels) };
        var z = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_full(&z, &shape, shape.len, zero, .float32, s));
        break :blk z;
    };
    const sh = mlx.getShape(buf);
    const start = [_]c_int{ 0, @intCast(tt.out_start), @intCast(ht.out_start), @intCast(wt.out_start), 0 };
    const stop = [_]c_int{ sh[0], @intCast(tt.out_end), @intCast(ht.out_end), @intCast(wt.out_end), sh[4] };
    const str = [_]c_int{ 1, 1, 1, 1, 1 };
    var cur = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cur);
    try mlx.check(mlx.mlx_slice(&cur, buf, &start, 5, &stop, 5, &str, 5, s));
    const summed = try addArr(cur, tile, s);
    defer _ = mlx.mlx_array_free(summed);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice_update(&out, buf, summed, &start, 5, &stop, 5, &str, 5, s));
    _ = mlx.mlx_array_free(buf);
    _ = mlx.mlx_array_eval(out);
    return out;
}

/// Crop the decoded volume back to the requested content shape: temporal pad is
/// always trailing, spatial pads are the recorded symmetric ones.
fn cropContent(x: mlx.mlx_array, content_px: [3]u32, h_before: u32, w_before: u32, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const start = [_]c_int{ 0, 0, @intCast(h_before), @intCast(w_before), 0 };
    const stop = [_]c_int{
        sh[0],
        @min(sh[1], @as(c_int, @intCast(content_px[0]))),
        @intCast(h_before + content_px[1]),
        @intCast(w_before + content_px[2]),
        sh[4],
    };
    const str = [_]c_int{ 1, 1, 1, 1, 1 };
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    try mlx.check(mlx.mlx_slice(&sliced, x, &start, 5, &stop, 5, &str, 5, s));
    return contig(sliced, s);
}

// ── tests ────────────────────────────────────────────────────────────────

const testing = std.testing;

test "the tile plan cuts only until one tile's diffusion volume fits" {
    // 768x512x97: 2.38M stage-5 tokens, one tile, no seams to blend.
    const s4_small = geom.stage4FromLatent(geom.production, .{ 13, 16, 24 }, true);
    const small = planTiles(geom.production, s4_small, MAX_TILE_TOKENS);
    try testing.expectEqual(s4_small, small.tile);

    // 1920x1088x97: 12.7M tokens — has to be cut, and every axis stays above
    // BOTH its NA floor and twice its seam overlap.
    const s4_big = geom.stage4FromLatent(geom.production, .{ 13, 34, 60 }, true);
    try testing.expectEqual([3]u32{ 49, 136, 240 }, s4_big);
    const big = planTiles(geom.production, s4_big, MAX_TILE_TOKENS);
    const up = geom.production.upsamples[3].stride;
    const m = geom.AxisMap{ .scale = up[0], .temporal = true };
    const tokens: u64 = @as(u64, m.map(big.tile[0])) * (big.tile[1] * up[1]) * (big.tile[2] * up[2]);
    try testing.expect(tokens <= MAX_TILE_TOKENS);
    const mins = geom.tileMinSize(geom.production);
    for (0..3) |a| {
        try testing.expect(big.tile[a] >= mins[a]);
        try testing.expect(big.tile[a] >= 2 * big.overlap[a]);
    }
    // The overlap is the larger of the stage-4 and diffusion-stage halos.
    try testing.expectEqual([3]u32{ 4, 12, 12 }, big.overlap);
}

test "a budget nothing can satisfy stops at the floors instead of looping" {
    const s4 = geom.stage4FromLatent(geom.production, .{ 13, 34, 60 }, true);
    const plan = planTiles(geom.production, s4, 1);
    const mins = geom.tileMinSize(geom.production);
    for (0..3) |a| {
        try testing.expect(plan.tile[a] >= @max(mins[a], 2 * plan.overlap[a]));
    }
}

// End-to-end smoke on the REAL checkpoint: the whole ladder runs, the shape is
// the one the geometry predicts, and nothing came back non-finite. Parity
// against the PyTorch reference is `tests/dump_ltx_diffvae_fixtures.py` + the
// gated oracle; this is the bring-up bisector.
//   LTX_DIFFVAE_MODEL = a pack dir holding vae_diffusion_decoder.safetensors
test "diffvae decode runs the whole ladder on the shipped checkpoint" {
    const dir = std.mem.span(std.c.getenv("LTX_DIFFVAE_MODEL") orelse return error.SkipZigTest);
    const alloc = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_s);

    const path = try std.fmt.allocPrintSentinel(alloc, "{s}/{s}", .{ dir, FILE_NAME }, 0);
    defer alloc.free(path);
    var comp = try load(alloc, path, cpu_s);
    defer comp.deinit();

    const latent_shape = [_]c_int{ 1, 128, 4, 8, 8 };
    const latent = try randomNormal(&latent_shape, 7, s);
    defer _ = mlx.mlx_array_free(latent);
    const lat_bf = try astype(latent, .bfloat16, s);
    defer _ = mlx.mlx_array_free(lat_bf);

    const px = try decode(alloc, &comp, geom.production, lat_bf, .{ .seed = 11 }, s);
    defer _ = mlx.mlx_array_free(px);
    const sh = mlx.getShape(px);
    const want = geom.pixelShape(geom.production, .{ 4, 8, 8 });
    try testing.expectEqual([_]c_int{ 1, 3, @intCast(want[0]), @intCast(want[1]), @intCast(want[2]) }, sh[0..5].*);

    const f32px = try astype(px, .float32, s);
    defer _ = mlx.mlx_array_free(f32px);
    _ = mlx.mlx_array_eval(f32px);
    const data = mlx.mlx_array_data_float32(f32px).?;
    const n: usize = mlx.mlx_array_size(f32px);
    var mean: f64 = 0;
    var absmax: f64 = 0;
    for (0..n) |i| {
        try testing.expect(std.math.isFinite(data[i]));
        mean += data[i];
        absmax = @max(absmax, @abs(data[i]));
    }
    std.debug.print("[diffvae-smoke] {any} mean={d:.4} absmax={d:.4}\n", .{ sh, mean / @as(f64, @floatFromInt(n)), absmax });
    // A decoder that produced a constant (or exploded) is not "running".
    try testing.expect(absmax > 0.01 and absmax < 100.0);
}

fn readF32(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]f32 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(allocator, .limited(1024 * 1024 * 1024));
    defer allocator.free(bytes);
    const cnt = bytes.len / 4;
    const out = try allocator.alloc(f32, cnt);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. cnt * 4]);
    return out;
}

/// Cosine AND rms ratio. A cosine alone cannot see a scale error, and this
/// stack concatenates a context volume with x — so a stage that came out
/// uniformly 2x would score a perfect 1.0 here and wreck the blocks reading it.
fn compareTo(name: []const u8, got: mlx.mlx_array, want: []const f32, cos_bar: f64, s: S) !void {
    const f32got = try astype(got, .float32, s);
    defer _ = mlx.mlx_array_free(f32got);
    _ = mlx.mlx_array_eval(f32got);
    const n: usize = mlx.mlx_array_size(f32got);
    try testing.expectEqual(want.len, n);
    const data = mlx.mlx_array_data_float32(f32got).?;
    var dot: f64 = 0;
    var sq_got: f64 = 0;
    var sq_want: f64 = 0;
    var max_abs: f64 = 0;
    for (0..n) |i| {
        const a: f64 = data[i];
        const b: f64 = want[i];
        dot += a * b;
        sq_got += a * a;
        sq_want += b * b;
        max_abs = @max(max_abs, @abs(a - b));
    }
    const cos = dot / (@sqrt(sq_got) * @sqrt(sq_want));
    const rms_ratio = @sqrt(sq_got) / @sqrt(sq_want);
    std.debug.print("[diffvae-parity] {s:<10} n={d:<9} cos={d:.6} rms_ratio={d:.4} max_abs={d:.4}\n", .{ name, n, cos, rms_ratio, max_abs });
    try testing.expect(cos > cos_bar);
    try testing.expect(rms_ratio > 0.98 and rms_ratio < 1.02);
}

// Per-stage parity against the PyTorch reference (tests/dump_ltx_diffvae_fixtures.py).
// Runs the SAME ungated path the oracle does — no ghost pad, no size floor, one
// tile — so a failure here is the MATH, and the tiling has its own guard.
//   LTX_DIFFVAE_MODEL + LTX_DIFFVAE_{LATENT,STAGE0,STAGE1,STAGE2,CONTEXT,XT,VPRED,PIXELS}
test "diffvae parity: every stage reproduces the reference decoder" {
    const dir = std.mem.span(std.c.getenv("LTX_DIFFVAE_MODEL") orelse return error.SkipZigTest);
    const lat_p = std.mem.span(std.c.getenv("LTX_DIFFVAE_LATENT") orelse return error.SkipZigTest);
    const alloc = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_s);

    const path = try std.fmt.allocPrintSentinel(alloc, "{s}/{s}", .{ dir, FILE_NAME }, 0);
    defer alloc.free(path);
    var comp = try load(alloc, path, cpu_s);
    defer comp.deinit();
    const cfg = geom.production;
    const c = Ctx{ .comp = &comp, .cfg = cfg, .alloc = alloc, .s = s };

    const latbuf = try readF32(io, alloc, lat_p);
    defer alloc.free(latbuf);
    const lat_shape = [_]c_int{ 1, 128, 3, 8, 8 };
    const lat_bcfhw = mlx.mlx_array_new_data(latbuf.ptr, &lat_shape, 5, .float32);
    defer _ = mlx.mlx_array_free(lat_bcfhw);
    const lat_hwc = try transposeTo(lat_bcfhw, &[_]c_int{ 0, 2, 3, 4, 1 }, s);
    defer _ = mlx.mlx_array_free(lat_hwc);

    var x = blk: {
        var mk: [128]u8 = undefined;
        const mean = try c.tensor(try std.fmt.bufPrint(&mk, "{s}.per_channel_statistics.mean", .{PREFIX}));
        var sk: [128]u8 = undefined;
        const stdv = try c.tensor(try std.fmt.bufPrint(&sk, "{s}.per_channel_statistics.std", .{PREFIX}));
        const scaled = try mulArr(lat_hwc, stdv, s);
        defer _ = mlx.mlx_array_free(scaled);
        const shifted = try addArr(scaled, mean, s);
        defer _ = mlx.mlx_array_free(shifted);
        var ck: [128]u8 = undefined;
        const conv_in = try std.fmt.bufPrint(&ck, "{s}.conv_in", .{PREFIX});
        const typed = try astype(shifted, mlx.mlx_array_dtype(try c.weight(conv_in)), s);
        defer _ = mlx.mlx_array_free(typed);
        break :blk try c.lin(typed, conv_in);
    };
    defer _ = mlx.mlx_array_free(x);

    const stage_env = [4][:0]const u8{ "LTX_DIFFVAE_STAGE0", "LTX_DIFFVAE_STAGE1", "LTX_DIFFVAE_STAGE2", "LTX_DIFFVAE_CONTEXT" };
    const stage_name = [4][]const u8{ "stage0", "stage1", "stage2", "context" };
    for (0..4) |i| {
        const nx = try detStage(c, x, @intCast(i), true);
        _ = mlx.mlx_array_free(x);
        x = nx;
        _ = mlx.mlx_array_eval(x);
        _ = mlx.mlx_clear_cache();
        if (std.c.getenv(stage_env[i].ptr)) |p| {
            const want = try readF32(io, alloc, std.mem.span(p));
            defer alloc.free(want);
            try compareTo(stage_name[i], x, want, 0.999, s);
        }
    }
    const context = x;

    const xt_p = std.mem.span(std.c.getenv("LTX_DIFFVAE_XT") orelse return error.SkipZigTest);
    const xtbuf = try readF32(io, alloc, xt_p);
    defer alloc.free(xtbuf);
    const csh = mlx.getShape(context);
    const xt_shape = [_]c_int{ 1, csh[1], csh[2], csh[3], @intCast(cfg.patchChannels()) };
    const x_t0 = mlx.mlx_array_new_data(xtbuf.ptr, &xt_shape, 5, .float32);
    defer _ = mlx.mlx_array_free(x_t0);

    var rope = try buildRope(alloc, cfg, .{ @intCast(csh[1]), @intCast(csh[2]), @intCast(csh[3]) }, s);
    defer rope.deinit();
    const model_dtype = mlx.mlx_array_dtype(context);

    if (std.c.getenv("LTX_DIFFVAE_VPRED")) |p| {
        const want = try readF32(io, alloc, std.mem.span(p));
        defer alloc.free(want);
        const x_in = try astype(x_t0, model_dtype, s);
        defer _ = mlx.mlx_array_free(x_in);
        // At the SHIPPED timestep: x1000 is what the checkpoint was trained on
        // (x1 decodes to static), so the modulation this exercises is the real one.
        const v = try diffStep(c, context, x_in, Sampler.default.timestep_scale, rope);
        defer _ = mlx.mlx_array_free(v);
        try compareTo("vpred", v, want, 0.999, s);
        _ = mlx.mlx_clear_cache();
    }

    if (std.c.getenv("LTX_DIFFVAE_PIXELS")) |p| {
        const want = try readF32(io, alloc, std.mem.span(p));
        defer alloc.free(want);
        // The shipped contract: ONE step, the prediction IS x0.
        var tbuf: [8]f32 = undefined;
        const ts = geom.timesteps(Sampler.default.num_steps, &tbuf);
        try testing.expectEqual(@as(usize, 1), ts.len);
        const x_in = try astype(x_t0, model_dtype, s);
        defer _ = mlx.mlx_array_free(x_in);
        const pred = try diffStep(c, context, x_in, ts[0] * Sampler.default.timestep_scale, rope);
        defer _ = mlx.mlx_array_free(pred);
        const x_t = try astype(pred, .float32, s);
        defer _ = mlx.mlx_array_free(x_t);
        const px_hwc = try unpatchify(x_t, cfg.patch_size, s);
        defer _ = mlx.mlx_array_free(px_hwc);
        const px_t = try transposeTo(px_hwc, &[_]c_int{ 0, 4, 1, 2, 3 }, s);
        defer _ = mlx.mlx_array_free(px_t);
        const px = try contig(px_t, s);
        defer _ = mlx.mlx_array_free(px);
        try compareTo("pixels", px, want, 0.999, s);
    }
}

// Tiling is a MEMORY measure, not a different decode: the same clip whole and
// cut into tiles must agree. A cosine over the whole frame will happily hide a
// visible seam, so the crop CENTRED on the seam is asserted separately — that
// is where a mismatched halo, an independently-drawn noise field or a blend
// that does not sum to 1 shows up.
//   LTX_DIFFVAE_MODEL = a pack dir holding vae_diffusion_decoder.safetensors
test "diffvae tiled decode agrees with the whole-volume decode, seams included" {
    const dir = std.mem.span(std.c.getenv("LTX_DIFFVAE_MODEL") orelse return error.SkipZigTest);
    const alloc = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_s);

    const path = try std.fmt.allocPrintSentinel(alloc, "{s}/{s}", .{ dir, FILE_NAME }, 0);
    defer alloc.free(path);
    var comp = try load(alloc, path, cpu_s);
    defer comp.deinit();

    const latent_shape = [_]c_int{ 1, 128, 3, 8, 8 };
    const latent = try randomNormal(&latent_shape, 3, s);
    defer _ = mlx.mlx_array_free(latent);
    const lat_bf = try astype(latent, .bfloat16, s);
    defer _ = mlx.mlx_array_free(lat_bf);

    // Same seed → the same global noise field on both arms, which is the whole
    // reason a tile may slice it rather than draw its own.
    // ENGAGEMENT: the tiled arm must actually be cut, or both arms ran the same
    // path and every number below agrees for the wrong reason.
    {
        const s4 = geom.stage4FromLatent(geom.production, .{ 3, 8, 8 }, true);
        const plan = planTiles(geom.production, s4, 20_000);
        const scale = geom.stage4PixelScale(geom.production);
        var buf: [MAX_TILES_PER_AXIS]geom.Interval = undefined;
        const nt = geom.splitAxis(s4[0], plan.tile[0], plan.overlap[0], 3, .{ .scale = scale[0], .temporal = true }, &buf).len;
        var buf2: [MAX_TILES_PER_AXIS]geom.Interval = undefined;
        const nw = geom.splitAxis(s4[2], plan.tile[2], plan.overlap[2], 5, .{ .scale = scale[2] }, &buf2).len;
        try testing.expect(nt > 1 and nw > 1);
    }

    const whole = try decode(alloc, &comp, geom.production, lat_bf, .{ .seed = 5 }, s);
    defer _ = mlx.mlx_array_free(whole);
    const tiled = try decode(alloc, &comp, geom.production, lat_bf, .{ .seed = 5, .tile_tokens = 20_000 }, s);
    defer _ = mlx.mlx_array_free(tiled);
    try testing.expectEqualSlices(c_int, mlx.getShape(whole), mlx.getShape(tiled));

    const a32 = try astype(whole, .float32, s);
    defer _ = mlx.mlx_array_free(a32);
    const b32 = try astype(tiled, .float32, s);
    defer _ = mlx.mlx_array_free(b32);
    _ = mlx.mlx_array_eval(a32);
    _ = mlx.mlx_array_eval(b32);
    const av = mlx.mlx_array_data_float32(a32).?;
    const bv = mlx.mlx_array_data_float32(b32).?;

    const sh = mlx.getShape(whole); // [1,3,F,H,W]
    const frames: usize = @intCast(sh[2]);
    const height: usize = @intCast(sh[3]);
    const width: usize = @intCast(sh[4]);

    // Whole-frame agreement, then the same statistic over a band centred on the
    // W seam (the tile plan cuts W first, so its ramp sits mid-frame).
    const Stat = struct {
        fn run(x: [*]const f32, y: [*]const f32, idx: []const usize) struct { cos: f64, rms: f64 } {
            var dot: f64 = 0;
            var sa: f64 = 0;
            var sb: f64 = 0;
            for (idx) |i| {
                dot += @as(f64, x[i]) * @as(f64, y[i]);
                sa += @as(f64, x[i]) * @as(f64, x[i]);
                sb += @as(f64, y[i]) * @as(f64, y[i]);
            }
            return .{ .cos = dot / (@sqrt(sa) * @sqrt(sb)), .rms = @sqrt(sa) / @sqrt(sb) };
        }
    };

    var all = std.ArrayList(usize){ .items = &.{}, .capacity = 0 };
    defer all.deinit(alloc);
    var seam = std.ArrayList(usize){ .items = &.{}, .capacity = 0 };
    defer seam.deinit(alloc);
    const w_lo = width / 2 - width / 8;
    const w_hi = width / 2 + width / 8;
    for (0..3) |ch| {
        for (0..frames) |f| {
            for (0..height) |h| {
                for (0..width) |w| {
                    const i = ((ch * frames + f) * height + h) * width + w;
                    try all.append(alloc, i);
                    if (w >= w_lo and w < w_hi) try seam.append(alloc, i);
                }
            }
        }
    }
    const whole_stat = Stat.run(av, bv, all.items);
    const seam_stat = Stat.run(av, bv, seam.items);
    std.debug.print(
        "[diffvae-tile] whole cos={d:.7} rms={d:.5} | seam band cos={d:.7} rms={d:.5}\n",
        .{ whole_stat.cos, whole_stat.rms, seam_stat.cos, seam_stat.rms },
    );
    // Measured 0.9999972 / 0.9999974 on this clip; the bar keeps headroom so a
    // bigger canvas's longer seams cannot make it knife-edge.
    try testing.expect(whole_stat.cos > 0.9995);
    try testing.expect(whole_stat.rms > 0.99 and whole_stat.rms < 1.01);
    try testing.expect(seam_stat.cos > 0.999);
    try testing.expect(seam_stat.rms > 0.99 and seam_stat.rms < 1.01);
}

// What the decode actually PEAKS at, at a production canvas. The residency bill
// promises a bound (`gen.ltxDiffVaeDecodeBytes`); this is the measurement that
// bound is set from, and the assertion that the tile budget still holds it.
// `MLX_SERVE_DIFFVAE_TILE_TOKENS` is the lever both read.
//   LTX_DIFFVAE_MODEL = a pack dir; ~768x512x97, minutes.
test "diffvae decode peak stays inside the budget the residency bill promises" {
    const dir = std.mem.span(std.c.getenv("LTX_DIFFVAE_MODEL") orelse return error.SkipZigTest);
    if (std.c.getenv("LTX_DIFFVAE_PEAK") == null) return error.SkipZigTest;
    const alloc = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_s);

    const path = try std.fmt.allocPrintSentinel(alloc, "{s}/{s}", .{ dir, FILE_NAME }, 0);
    defer alloc.free(path);
    var comp = try load(alloc, path, cpu_s);
    defer comp.deinit();

    const latent_shape = [_]c_int{ 1, 128, 13, 16, 24 }; // 97f at 768x512
    const latent = try randomNormal(&latent_shape, 1, s);
    defer _ = mlx.mlx_array_free(latent);
    const lat_bf = try astype(latent, .bfloat16, s);
    defer _ = mlx.mlx_array_free(lat_bf);
    _ = mlx.mlx_array_eval(lat_bf);

    var before: usize = 0;
    _ = mlx.mlx_get_active_memory(&before);
    _ = mlx.mlx_reset_peak_memory();
    const io = std.Io.Threaded.global_single_threaded.io();
    var timer = io_util.Stopwatch.init(io);
    const px = try decode(alloc, &comp, geom.production, lat_bf, .{ .seed = 2 }, s);
    const ms = timer.read() / 1_000_000;
    defer _ = mlx.mlx_array_free(px);
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    const gb = @as(f64, @floatFromInt(peak -| before)) / (1024.0 * 1024.0 * 1024.0);
    std.debug.print("[diffvae-peak] {any} in {d} ms, peak +{d:.2} GiB (budget {d} tokens/tile)\n", .{ mlx.getShape(px), ms, gb, tileTokenBudget() });
    try testing.expect(gb < @as(f64, @floatFromInt(DECODE_PEAK_BUDGET_GIB)));
}

test "the tile budget follows free memory and never leaves the measured band" {
    const GB: u64 = 1024 * 1024 * 1024;
    // A failed memory query must not stop a decode — take the ceiling.
    try testing.expectEqual(MAX_TILE_TOKENS, tileTokensForMemory(0));
    // A big Mac with room reaches the ceiling (one tile at 768x512x97).
    try testing.expectEqual(MAX_TILE_TOKENS, tileTokensForMemory(64 * GB));
    // A tight machine cuts: 12 GiB free → half of it over 10 KiB/token.
    const tight = tileTokensForMemory(12 * GB);
    try testing.expect(tight > MIN_TILE_TOKENS and tight < MAX_TILE_TOKENS);
    try testing.expectEqual((12 * GB / 100 * 50) / BYTES_PER_STAGE5_TOKEN, tight);
    // And a starved one still decodes rather than planning a zero-size tile.
    try testing.expectEqual(MIN_TILE_TOKENS, tileTokensForMemory(1 * GB));
}

test "the sampler contract is the measured one, not the reference's class defaults" {
    // No pack ships a `vae` config, so `model_output_type`, the step count and
    // `timestep_scale_multiplier` are constructor arguments nobody declares.
    // Taking the reference class defaults (v-prediction, 2 steps, x1) decodes
    // the shipped weights to STATIC — measured adjacent-pixel gradient 30.2 at
    // x1, 17.9 at 2 steps and 44.4 at both, against 2.4 for the arm below and
    // 2.2 for the conv decoder on the same clip. This pins the arm that
    // produces a picture, so "restoring the reference defaults" is a red test
    // with a reason rather than a silently noisy decode.
    try testing.expect(Sampler.default.predict_x0);
    try testing.expectEqual(@as(u32, 1), Sampler.default.num_steps);
    try testing.expectApproxEqAbs(@as(f32, 1000.0), Sampler.default.timestep_scale, 1e-6);
    // One step means the schedule is a single t=1.0, and the prediction IS the
    // answer — there is no Euler update left to take.
    var buf: [4]f32 = undefined;
    const ts = geom.timesteps(Sampler.default.num_steps, &buf);
    try testing.expectEqual(@as(usize, 1), ts.len);
    try testing.expectApproxEqAbs(@as(f32, 1.0), ts[0], 1e-6);
}
