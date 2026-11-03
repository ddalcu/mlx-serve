//! Hyper-connection reads and writes fused into custom kernels (the deferred write folded into the next read, the
//! prefill-width pending write), with their kill switches.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const hc_decode2 = @import("../../sdk/blocks/hc/hc_decode2.zig");
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const testing = @import("../../engines/mlx/kv/cache.zig").testing;
const Trunk = @import("trunk.zig").Trunk;
const attn256MaxDiff = @import("../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const qmatmulBits = @import("../../sdk/blocks/linear/qmatmul.zig").qmatmulBits;
const verifySharedHardware = @import("../../sdk/blocks/linear/verify_projection.zig").verifySharedHardware;
const standinOnes = @import("../../sdk/diag/standin.zig").standinOnes;
const standinRef = @import("../../sdk/diag/standin.zig").standinRef;
const bf16BitEqual = @import("../../sdk/testing/parity.zig").bf16BitEqual;
const testReadF32 = @import("../../sdk/testing/parity.zig").testReadF32;
const HcWeights = @import("weights.zig").HcWeights;

// ── qwen4_exp fused hyper-connection READ at decode and verify width ──
// The read is two launches (down, then up/mix) in hc_decode2.zig for 1..HC_FUSED_MAX_ROWS rows; anything outside its
// geometry keeps the unfused chain.
var hc_fused_eps: ?mlx.mlx_array = null;
/// Test seam: false keeps the unfused chain.
pub var hc_fused_override: ?bool = null;
var hc_fused_eps_val: f32 = 0;

pub fn hcFusedEnabled() bool {
    return hc_fused_override orelse true;
}

pub const HcFusedOut = struct { mixed: mlx.mlx_array, inj: mlx.mlx_array, stream: mlx.mlx_array };

/// A deferred `hcWrite`: the next read's N kernel applies `stream + out*inj`
/// itself (one dispatch fewer per block). Handles are retained copies.
pub const HcPending = struct {
    out: mlx.mlx_array, // [B,S,hidden]
    inj: mlx.mlx_array, // [B,S,hc,1]
    pub fn deinit(self: *HcPending) void {
        _ = mlx.mlx_array_free(self.out);
        _ = mlx.mlx_array_free(self.inj);
    }
};

/// Fused decode-width hyper-connection read over `batch*seq` rows (1..
/// HC_FUSED_MAX_ROWS: decode, verify widths, batched slots — rows are
/// independent, the grid carries them, weights are read once per row).
/// `x` holds `batch*seq*hc*hidden` elements; `iw` is the row-major dense
/// `[hc*hidden, hc]` inject weight or null-ctx (mixer). With `pend`, the read
/// first applies that pending write and returns the written stream in
/// `.stream` (null-ctx otherwise). Returns `mixed [B,S,hidden]` +
/// `inj [B,S,hc,1]`, or null when the geometry/quant is outside the kernel
/// (caller keeps the chain).
pub const HC_FUSED_MAX_ROWS: c_int = 16;
pub fn hcReadJoined(a: std.mem.Allocator, s: mlx.mlx_stream, inputs: []const mlx.mlx_array, w: HcWeights, width: c_int, eps: f32, bits: u32, gs: u32, pending: ?[]const HcPending) !?[]HcFusedOut {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or inputs.len < 2 or inputs.len > 8 or width < 2) return null;
    const total = @as(c_int, @intCast(inputs.len)) * width;
    if (total > HC_FUSED_MAX_ROWS or w.inject_flat.ctx == null) return null;
    for (inputs) |input| {
        if (input.ctx == null or mlx.mlx_array_dtype(input) != .bfloat16 or !std.mem.eql(c_int, mlx.getShape(input), &.{ 1, width, 10240 })) return null;
    }
    if (pending) |values| {
        if (values.len != inputs.len) return null;
        for (values) |p| {
            if (p.out.ctx == null or p.inj.ctx == null or mlx.mlx_array_dtype(p.out) != .bfloat16 or mlx.mlx_array_dtype(p.inj) != .bfloat16 or
                !std.mem.eql(c_int, mlx.getShape(p.out), &.{ 1, width, 2560 }) or !std.mem.eql(c_int, mlx.getShape(p.inj), &.{ 1, width, 4, 1 })) return null;
        }
    }
    const Join = struct {
        fn array(st: mlx.mlx_stream, values: []const mlx.mlx_array) !mlx.mlx_array {
            const vec = mlx.mlx_vector_array_new_data(values.ptr, values.len);
            defer _ = mlx.mlx_vector_array_free(vec);
            var out = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, st));
            return out;
        }
    };
    const joined = try Join.array(s, inputs);
    defer _ = mlx.mlx_array_free(joined);
    var joined_pending: ?HcPending = null;
    defer if (joined_pending) |p| {
        _ = mlx.mlx_array_free(p.out);
        _ = mlx.mlx_array_free(p.inj);
    };
    if (pending) |values| {
        var outs: [8]mlx.mlx_array = undefined;
        var injs: [8]mlx.mlx_array = undefined;
        for (values, 0..) |p, i| {
            outs[i] = p.out;
            injs[i] = p.inj;
        }
        const out = try Join.array(s, outs[0..values.len]);
        errdefer _ = mlx.mlx_array_free(out);
        joined_pending = .{ .out = out, .inj = try Join.array(s, injs[0..values.len]) };
    }
    const dense = w.down_rows.ctx != null;
    const full = (try hcReadFused(s, joined, 1, total, w.norm_w, if (dense) w.down_rows else w.down_w, w.down_s, w.down_b, if (dense) w.up_rows else w.up_w, w.up_s, w.up_b, w.inject_flat, eps, 4, 2560, bits, gs, joined_pending)) orelse return null;
    defer inline for (.{ "mixed", "inj", "stream" }) |name| {
        if (@field(full, name).ctx != null) _ = mlx.mlx_array_free(@field(full, name));
    };
    const result = try a.alloc(HcFusedOut, inputs.len);
    @memset(result, .{ .mixed = .{}, .inj = .{}, .stream = .{} });
    errdefer {
        for (result) |r| inline for (.{ "mixed", "inj", "stream" }) |name| {
            if (@field(r, name).ctx != null) _ = mlx.mlx_array_free(@field(r, name));
        };
        a.free(result);
    }
    for (result, 0..) |*r, i| inline for (.{ "mixed", "inj", "stream" }) |name| {
        const value = @field(full, name);
        if (value.ctx != null) {
            const shape = mlx.getShape(value);
            var start: [4]c_int = @splat(0);
            var end: [4]c_int = @splat(1);
            @memcpy(end[0..shape.len], shape);
            start[1] = @as(c_int, @intCast(i)) * width;
            end[1] = start[1] + width;
            @field(r, name) = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&@field(r, name), value, &start, shape.len, &end, shape.len, &[_]c_int{ 1, 1, 1, 1 }, shape.len, s));
        }
    };
    return result;
}

/// The epsilon operand the hc kernels bind, cached per value.
fn hcEpsArray(eps: f32) !mlx.mlx_array {
    if (hc_fused_eps == null or hc_fused_eps_val != eps) {
        if (hc_fused_eps) |e| _ = mlx.mlx_array_free(e);
        const esh = [_]c_int{1};
        var ev = eps;
        hc_fused_eps = mlx.mlx_array_new_data(&ev, &esh, 1, .float32);
        hc_fused_eps_val = eps;
    }
    return hc_fused_eps.?;
}

pub fn hcReadFused(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    batch: c_int,
    seq: c_int,
    nw: mlx.mlx_array,
    dw: mlx.mlx_array,
    ds: mlx.mlx_array,
    db: mlx.mlx_array,
    uw: mlx.mlx_array,
    us: mlx.mlx_array,
    ub: mlx.mlx_array,
    iw: mlx.mlx_array,
    eps: f32,
    hc: c_int,
    hidden: c_int,
    bits: u32,
    group_size: u32,
    pend: ?HcPending,
) !?HcFusedOut {
    if (!hcFusedEnabled()) return null;
    if (!mlx.streamIsGpu(s) and !mlx.cudaAvailable()) return null;
    const vpw = hc_decode2.packValues(bits) orelse return null;
    const rows = batch * seq;
    if (rows < 1 or rows > HC_FUSED_MAX_ROWS) return null;
    if (@rem(@as(c_int, @intCast(group_size)), vpw) != 0) return null;
    if (bits != 16 and (ds.ctx == null or db.ctx == null or us.ctx == null or ub.ctx == null)) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(nw) != xd) return null;
    const K: c_int = hc * hidden;
    if (hc < 1 or hc > 8 or hidden < 8 or @rem(hidden, 8) != 0 or @rem(hidden, vpw) != 0) return null;
    if (mlx.mlx_array_size(x) != @as(usize, @intCast(rows * K)) or mlx.mlx_array_size(nw) != @as(usize, @intCast(K))) return null;
    const dsh = mlx.getShape(dw);
    const ush = mlx.getShape(uw);
    if (dsh.len != 2 or ush.len != 2) return null;
    const R: c_int = dsh[0];
    const bi: c_int = @intCast(bits);
    if (dsh[1] * 32 != K * bi) return null;
    if (ush[0] != K or ush[1] * 32 != R * bi) return null;
    const gsi: c_int = @intCast(group_size);
    if (@rem(K, gsi) != 0 or @rem(R, gsi) != 0) return null;
    // N: 256 threads per stream; D: 8 simdgroups × 32 lanes split each row's packs.
    if (@rem(hidden, 256) != 0 or @rem(@divTrunc(K, vpw), 256) != 0 or @rem(R, vpw) != 0) return null;
    const inj: c_int = @intFromBool(iw.ctx != null);
    if (pend) |pd| {
        if (mlx.mlx_array_dtype(pd.out) != xd or mlx.mlx_array_dtype(pd.inj) != xd) return null;
        if (mlx.mlx_array_size(pd.out) != @as(usize, @intCast(rows * hidden)) or mlx.mlx_array_size(pd.inj) != @as(usize, @intCast(rows * hc))) return null;
    }
    if (inj == 1) {
        if (mlx.mlx_array_dtype(iw) != xd) return null;
        const ish = mlx.getShape(iw);
        if (ish.len != 2 or ish[0] != K or ish[1] != hc) return null;
    }

    const pending: ?hc_decode2.Pending = if (pend) |pd| .{ .out = pd.out, .inj = pd.inj } else null;
    const read = if (mlx.cudaAvailable())
        try @import("../../sdk/blocks/hc/hc_cuda.zig").read(s, x, nw, dw, ds, db, uw, us, ub, iw, try hcEpsArray(eps), rows, hc, hidden, bits, group_size, pending)
    else if (rows <= hc_decode2.maxRows())
        try hc_decode2.read(s, x, nw, dw, ds, db, uw, us, ub, iw, try hcEpsArray(eps), rows, hc, hidden, bits, group_size, pending)
    else
        null;
    {
        if (read) |two| {
            defer inline for (.{ "mixed", "inj", "stream" }) |name| {
                if (@field(two, name).ctx != null) _ = mlx.mlx_array_free(@field(two, name));
            };
            const xsh = mlx.getShape(x);
            var out: HcFusedOut = .{ .mixed = mlx.mlx_array_new(), .inj = .{ .ctx = null }, .stream = .{ .ctx = null } };
            errdefer inline for (.{ "mixed", "inj", "stream" }) |name| {
                if (@field(out, name).ctx != null) _ = mlx.mlx_array_free(@field(out, name));
            };
            try mlx.check(mlx.mlx_reshape(&out.mixed, two.mixed, &[_]c_int{ batch, seq, hidden }, 3, s));
            if (two.inj.ctx != null) {
                out.inj = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_reshape(&out.inj, two.inj, &[_]c_int{ batch, seq, hc, 1 }, 4, s));
            }
            if (two.stream.ctx != null) {
                out.stream = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_reshape(&out.stream, two.stream, xsh.ptr, @intCast(xsh.len), s));
            }
            return out;
        }
    }
    return null;
}

// ── tests ──

test "fused hyper-connection read matches the op chain per element, at every packed width and dense bf16" {
    for ([_]u32{ 4, 5, 6, 8 }) |bits| for ([_]u32{ 64, 128 }) |gs| try hcFusedReadParity(bits, gs);
    try hcFusedReadParity(16, 64);
}

fn hcFusedReadParity(bits: u32, gs: u32) !void {
    // Reference = the unfused hcRead chain spelled in raw ops (grouped rms
    // norm with a ones weight, ×w, down qmm, silu, up qmm, sigmoid·n4 mean;
    // inject matvec, 2·sigmoid). Accumulation order differs, so the bar is
    // per-element within a few bf16 ulps, on every output of both kernels.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x4C0FFEE);
    const rnd = prng.random();
    hc_fused_override = true;
    defer hc_fused_override = null;

    const HC: c_int = 4;
    const H: c_int = 512;
    const K: c_int = HC * H;
    const R: c_int = 128;
    const eps: f32 = 1e-6;

    const Q = struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array };
    const quantRandom = struct {
        fn f(a: std.mem.Allocator, r: std.Random, st: mlx.mlx_stream, rows: c_int, cols: c_int, qbits: u32, qgs: u32) !Q {
            const buf = try a.alloc(f32, @intCast(rows * cols));
            defer a.free(buf);
            for (buf) |*v| v.* = (r.float(f32) - 0.5) * 0.2;
            const sh = [_]c_int{ rows, cols };
            const w32 = mlx.mlx_array_new_data(buf.ptr, &sh, 2, .float32);
            defer _ = mlx.mlx_array_free(w32);
            var wb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wb);
            try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, st));
            if (qbits == 16) {
                // Dense: the kernel reads the bf16 [rows, cols] bytes as words; the chain multiplies wb^T.
                var q: Q = .{ .w = mlx.mlx_array_new(), .sc = .{ .ctx = null }, .bi = mlx.mlx_array_new() };
                try mlx.check(mlx.mlx_view(&q.w, wb, .uint32, st));
                try mlx.check(mlx.mlx_transpose(&q.bi, wb, st));
                return q;
            }
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(qgs)), mlx.mlx_optional_int.some(@intCast(qbits)), "affine", .{}, st));
            var q: Q = undefined;
            q.w = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&q.w, triple, 0));
            q.sc = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&q.sc, triple, 1));
            q.bi = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&q.bi, triple, 2));
            return q;
        }
    }.f;
    const bf16Random = struct {
        fn f(a: std.mem.Allocator, r: std.Random, st: mlx.mlx_stream, shape: []const c_int, scale: f32, offset: f32) !mlx.mlx_array {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const buf = try a.alloc(f32, n);
            defer a.free(buf);
            for (buf) |*v| v.* = (r.float(f32) - 0.5) * scale + offset;
            const a32 = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(a32);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&out, a32, .bfloat16, st));
            return out;
        }
    }.f;

    const down = try quantRandom(allocator, rnd, s, R, K, bits, gs);
    defer {
        _ = mlx.mlx_array_free(down.w);
        _ = mlx.mlx_array_free(down.sc);
        _ = mlx.mlx_array_free(down.bi);
    }
    const up = try quantRandom(allocator, rnd, s, K, R, bits, gs);
    defer {
        _ = mlx.mlx_array_free(up.w);
        _ = mlx.mlx_array_free(up.sc);
        _ = mlx.mlx_array_free(up.bi);
    }
    const x = try bf16Random(allocator, rnd, s, &.{ 1, 1, K }, 4.0, 0.0);
    defer _ = mlx.mlx_array_free(x);
    const nw = try bf16Random(allocator, rnd, s, &.{ HC, H }, 1.0, 1.0);
    defer _ = mlx.mlx_array_free(nw);
    const iw = try bf16Random(allocator, rnd, s, &.{ K, HC }, 0.1, 0.0);
    defer _ = mlx.mlx_array_free(iw);
    const ones = try bf16Random(allocator, rnd, s, &.{H}, 0.0, 1.0);
    defer _ = mlx.mlx_array_free(ones);

    // Reference chain.
    const shape4 = [_]c_int{ 1, 1, HC, H };
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_reshape(&x4, x, &shape4, 4, s));
    var n4raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n4raw);
    try mlx.check(mlx.mlx_fast_rms_norm(&n4raw, x4, ones, eps, s));
    var n4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n4);
    try mlx.check(mlx.mlx_multiply(&n4, n4raw, nw, s));
    const flat_shape = [_]c_int{ 1, 1, K };
    var n_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n_flat);
    try mlx.check(mlx.mlx_reshape(&n_flat, n4, &flat_shape, 3, s));
    const d = if (bits == 16) blk: {
        var m = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&m, n_flat, down.bi, s));
        break :blk m;
    } else try qmatmulBits(n_flat, down.w, down.sc, down.bi, bits, gs, .affine, s);
    defer _ = mlx.mlx_array_free(d);
    var dsig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dsig);
    try mlx.check(mlx.mlx_sigmoid(&dsig, d, s));
    var act = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_multiply(&act, d, dsig, s));
    const u = if (bits == 16) blk: {
        var m = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&m, act, up.bi, s));
        break :blk m;
    } else try qmatmulBits(act, up.w, up.sc, up.bi, bits, gs, .affine, s);
    defer _ = mlx.mlx_array_free(u);
    var uu4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(uu4);
    try mlx.check(mlx.mlx_reshape(&uu4, u, &shape4, 4, s));
    var usig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(usig);
    try mlx.check(mlx.mlx_sigmoid(&usig, uu4, s));
    var prod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prod);
    try mlx.check(mlx.mlx_multiply(&prod, usig, n4, s));
    var ref_mixed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref_mixed);
    try mlx.check(mlx.mlx_mean_axis(&ref_mixed, prod, 2, false, s));
    var iraw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(iraw);
    try mlx.check(mlx.mlx_matmul(&iraw, n_flat, iw, s));
    var isig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(isig);
    try mlx.check(mlx.mlx_sigmoid(&isig, iraw, s));
    const two = mlx.mlx_array_new_float(2.0);
    defer _ = mlx.mlx_array_free(two);
    var ref_inj = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref_inj);
    try mlx.check(mlx.mlx_multiply(&ref_inj, isig, two, s));

    const got = (try hcReadFused(s, x, 1, 1, nw, down.w, down.sc, if (bits == 16) .{ .ctx = null } else down.bi, up.w, up.sc, if (bits == 16) .{ .ctx = null } else up.bi, iw, eps, HC, H, bits, gs, null)) orelse return error.HcFusedDeclined;
    defer {
        _ = mlx.mlx_array_free(got.mixed);
        _ = mlx.mlx_array_free(got.inj);
    }

    const checkClose = struct {
        fn f(a: std.mem.Allocator, st: mlx.mlx_stream, ref: mlx.mlx_array, val: mlx.mlx_array, n: usize) !void {
            const rh = try a.alloc(f32, n);
            defer a.free(rh);
            const gh = try a.alloc(f32, n);
            defer a.free(gh);
            try testReadF32(ref, rh, st);
            try testReadF32(val, gh, st);
            var worst: f32 = 0;
            var wr: f32 = 0;
            var wg: f32 = 0;
            var dot: f64 = 0;
            var nr: f64 = 0;
            var ng: f64 = 0;
            for (rh, gh) |r, g| {
                dot += @as(f64, r) * g;
                nr += @as(f64, r) * r;
                ng += @as(f64, g) * g;
                // Outputs are means/gates of O(1) bf16 terms: the floor is
                // one bf16 ulp at 1.0, so re-ordered accumulation of a
                // cancelling mean cannot fail on a small output.
                const tol = 0.02 * @abs(r) + 0.008;
                const diff = @abs(r - g);
                if (diff > tol and diff / tol > worst) {
                    worst = diff / tol;
                    wr = r;
                    wg = g;
                }
            }
            std.debug.print("fused hc read: n={d} cos={d:.6} worst diff/tol={d:.2} ref={d:.5} got={d:.5}\n", .{ n, dot / @sqrt(nr * ng), worst, wr, wg });
            if (worst > 0 or dot / @sqrt(nr * ng) < 0.9999) return error.HcFusedParity;
        }
    }.f;
    try checkClose(allocator, s, ref_mixed, got.mixed, @intCast(H));
    try checkClose(allocator, s, ref_inj, got.inj, @intCast(HC));

    // Pending-write arm: the read applies `x + out*inj` first and hands the
    // written stream back; feeding that stream to the plain read must agree.
    const wo = try bf16Random(allocator, rnd, s, &.{ 1, 1, H }, 2.0, 0.0);
    defer _ = mlx.mlx_array_free(wo);
    const wi = try bf16Random(allocator, rnd, s, &.{ 1, 1, HC, 1 }, 1.0, 1.0);
    defer _ = mlx.mlx_array_free(wi);
    var out4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out4);
    const o4 = [_]c_int{ 1, 1, 1, H };
    try mlx.check(mlx.mlx_reshape(&out4, wo, &o4, 4, s));
    var prod4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prod4);
    try mlx.check(mlx.mlx_multiply(&prod4, out4, wi, s));
    var sum4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum4);
    try mlx.check(mlx.mlx_add(&sum4, x4, prod4, s));
    var ref_stream = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref_stream);
    try mlx.check(mlx.mlx_reshape(&ref_stream, sum4, &flat_shape, 3, s));
    const plain = (try hcReadFused(s, ref_stream, 1, 1, nw, down.w, down.sc, if (bits == 16) .{ .ctx = null } else down.bi, up.w, up.sc, if (bits == 16) .{ .ctx = null } else up.bi, iw, eps, HC, H, bits, gs, null)) orelse return error.HcFusedDeclined;
    defer {
        _ = mlx.mlx_array_free(plain.mixed);
        _ = mlx.mlx_array_free(plain.inj);
    }
    const pend = (try hcReadFused(s, x, 1, 1, nw, down.w, down.sc, if (bits == 16) .{ .ctx = null } else down.bi, up.w, up.sc, if (bits == 16) .{ .ctx = null } else up.bi, iw, eps, HC, H, bits, gs, .{ .out = wo, .inj = wi })) orelse return error.HcFusedDeclined;
    defer {
        _ = mlx.mlx_array_free(pend.mixed);
        _ = mlx.mlx_array_free(pend.inj);
        _ = mlx.mlx_array_free(pend.stream);
    }
    try testing.expect(pend.stream.ctx != null);
    try checkClose(allocator, s, ref_stream, pend.stream, @intCast(K));
    try checkClose(allocator, s, plain.mixed, pend.mixed, @intCast(H));
    try checkClose(allocator, s, plain.inj, pend.inj, @intCast(HC));

    // Multi-row arm (verify widths / batched slots): the 3-row read is
    // BIT-identical to the three single-row reads stacked, plain and
    // pending, so an N>1 forward never changes what the N=1 path computed.
    const x3 = try bf16Random(allocator, rnd, s, &.{ 1, 3, K }, 4.0, 0.0);
    defer _ = mlx.mlx_array_free(x3);
    const wo3 = try bf16Random(allocator, rnd, s, &.{ 1, 3, H }, 2.0, 0.0);
    defer _ = mlx.mlx_array_free(wo3);
    const wi3 = try bf16Random(allocator, rnd, s, &.{ 1, 3, HC, 1 }, 1.0, 1.0);
    defer _ = mlx.mlx_array_free(wi3);
    const rows3 = (try hcReadFused(s, x3, 1, 3, nw, down.w, down.sc, if (bits == 16) .{ .ctx = null } else down.bi, up.w, up.sc, if (bits == 16) .{ .ctx = null } else up.bi, iw, eps, HC, H, bits, gs, .{ .out = wo3, .inj = wi3 })) orelse return error.HcFusedDeclined;
    defer {
        _ = mlx.mlx_array_free(rows3.mixed);
        _ = mlx.mlx_array_free(rows3.inj);
        _ = mlx.mlx_array_free(rows3.stream);
    }
    const expectRowsExact = struct {
        fn f(a: std.mem.Allocator, st: mlx.mlx_stream, stacked: mlx.mlx_array, single: mlx.mlx_array, row: usize, n: usize) !void {
            const all = try a.alloc(f32, n * 3);
            defer a.free(all);
            const one = try a.alloc(f32, n);
            defer a.free(one);
            try testReadF32(stacked, all, st);
            try testReadF32(single, one, st);
            for (0..n) |i| if (all[row * n + i] != one[i]) {
                std.debug.print("fused hc read rows: row {d} elem {d}: {d} vs {d}\n", .{ row, i, all[row * n + i], one[i] });
                return error.HcFusedRowsNotBitIdentical;
            };
        }
    }.f;
    for (0..3) |r| {
        const rc: c_int = @intCast(r);
        const st3 = [_]c_int{ 1, 1, 1 };
        var xr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xr);
        try mlx.check(mlx.mlx_slice(&xr, x3, &[_]c_int{ 0, rc, 0 }, 3, &[_]c_int{ 1, rc + 1, K }, 3, &st3, 3, s));
        var wor = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wor);
        try mlx.check(mlx.mlx_slice(&wor, wo3, &[_]c_int{ 0, rc, 0 }, 3, &[_]c_int{ 1, rc + 1, H }, 3, &st3, 3, s));
        var wir = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wir);
        const st4 = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&wir, wi3, &[_]c_int{ 0, rc, 0, 0 }, 4, &[_]c_int{ 1, rc + 1, HC, 1 }, 4, &st4, 4, s));
        const one = (try hcReadFused(s, xr, 1, 1, nw, down.w, down.sc, if (bits == 16) .{ .ctx = null } else down.bi, up.w, up.sc, if (bits == 16) .{ .ctx = null } else up.bi, iw, eps, HC, H, bits, gs, .{ .out = wor, .inj = wir })) orelse return error.HcFusedDeclined;
        defer {
            _ = mlx.mlx_array_free(one.mixed);
            _ = mlx.mlx_array_free(one.inj);
            _ = mlx.mlx_array_free(one.stream);
        }
        try expectRowsExact(allocator, s, rows3.mixed, one.mixed, r, @intCast(H));
        try expectRowsExact(allocator, s, rows3.inj, one.inj, r, @intCast(HC));
        try expectRowsExact(allocator, s, rows3.stream, one.stream, r, @intCast(K));
    }
    try testing.expect((try hcReadFused(s, x3, 1, 17, nw, down.w, down.sc, if (bits == 16) .{ .ctx = null } else down.bi, up.w, up.sc, if (bits == 16) .{ .ctx = null } else up.bi, iw, eps, HC, H, bits, gs, null)) == null);

    if (bits != 8 or gs != 64) return;
    // 8-bit rows take the vectorized up launch at every row count (an evaluated, aligned up weight; a lazy one has no data
    // pointer and keeps the per-row launch): every row matches the single-row read, bit for bit.
    const R8: c_int = 320; // production width: several uint4 blocks per lane, scale group changing between them
    const down8 = try quantRandom(allocator, rnd, s, R8, K, 8, 64);
    defer {
        _ = mlx.mlx_array_free(down8.w);
        _ = mlx.mlx_array_free(down8.sc);
        _ = mlx.mlx_array_free(down8.bi);
    }
    const up8 = try quantRandom(allocator, rnd, s, K, R8, 8, 64);
    defer {
        _ = mlx.mlx_array_free(up8.w);
        _ = mlx.mlx_array_free(up8.sc);
        _ = mlx.mlx_array_free(up8.bi);
    }
    for ([_]c_int{ 2, 6, 9, 16 }) |rows| {
        const x8 = try bf16Random(allocator, rnd, s, &.{ 1, rows, K }, 4.0, 0.0);
        defer _ = mlx.mlx_array_free(x8);
        const served_before = hc_decode2.served_vec;
        const many = (try hcReadFused(s, x8, 1, rows, nw, down8.w, down8.sc, down8.bi, up8.w, up8.sc, up8.bi, iw, eps, HC, H, 8, gs, null)) orelse return error.HcFusedDeclined;
        defer {
            _ = mlx.mlx_array_free(many.mixed);
            _ = mlx.mlx_array_free(many.inj);
        }
        try testing.expectEqual(served_before + 1, hc_decode2.served_vec);
        var r: c_int = 0;
        while (r < rows) : (r += 1) {
            const st3 = [_]c_int{ 1, 1, 1 };
            var xr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xr);
            try mlx.check(mlx.mlx_slice(&xr, x8, &[_]c_int{ 0, r, 0 }, 3, &[_]c_int{ 1, r + 1, K }, 3, &st3, 3, s));
            const one = (try hcReadFused(s, xr, 1, 1, nw, down8.w, down8.sc, down8.bi, up8.w, up8.sc, up8.bi, iw, eps, HC, H, 8, gs, null)) orelse return error.HcFusedDeclined;
            defer {
                _ = mlx.mlx_array_free(one.mixed);
                _ = mlx.mlx_array_free(one.inj);
            }
            var mrow = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(mrow);
            try mlx.check(mlx.mlx_slice(&mrow, many.mixed, &[_]c_int{ 0, r, 0 }, 3, &[_]c_int{ 1, r + 1, H }, 3, &st3, 3, s));
            var irow = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(irow);
            const st4 = [_]c_int{ 1, 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&irow, many.inj, &[_]c_int{ 0, r, 0, 0 }, 4, &[_]c_int{ 1, r + 1, HC, 1 }, 4, &st4, 4, s));
            inline for (.{ .{ one.mixed, mrow, H }, .{ one.inj, irow, HC } }) |pair| {
                const n: usize = @intCast(pair[2]);
                const left = try allocator.alloc(f32, n);
                defer allocator.free(left);
                const right = try allocator.alloc(f32, n);
                defer allocator.free(right);
                try testReadF32(pair[0], left, s);
                try testReadF32(pair[1], right, s);
                try testing.expectEqualSlices(f32, left, right);
            }
        }
    }
}

test "hc prefill: pending write, normalized streams, native inject reduction and BF16 mix across shapes" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const hp = @import("../../sdk/blocks/hc/hc_prefill.zig");
    const s = mlx.gpuStream();
    var xfm = std.mem.zeroInit(Trunk, .{ .s = s, .allocator = testing.allocator });
    xfm.compileQwen4Hc();
    defer inline for (.{ xfm.compiled_hc_silu, xfm.compiled_hc_mix, xfm.compiled_hc_inj, xfm.compiled_hc_write }) |slot| {
        if (slot) |closure| _ = mlx.mlx_closure_free(closure);
    };
    try testing.expect(xfm.compiled_hc_write != null and xfm.compiled_hc_mix != null);
    var prng = std.Random.DefaultPrng.init(0x91831);
    const rnd = prng.random();
    const eps = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps);
    for ([_][4]c_int{ .{ 2, 17, 4, 2560 }, .{ 1, 65, 4, 2560 }, .{ 1, 513, 4, 2560 }, .{ 1, 17, 1, 128 }, .{ 1, 33, 2, 1536 }, .{ 1, 33, 3, 1536 }, .{ 1, 33, 5, 1536 }, .{ 1, 33, 6, 1536 }, .{ 1, 33, 7, 1536 }, .{ 1, 33, 8, 4096 }, .{ 1, 8703, 4, 2560 } }) |shape| {
        const b = shape[0];
        const seq = shape[1];
        xfm.config.hc_count = @intCast(shape[2]);
        xfm.config.hidden_size = @intCast(shape[3]);
        const hc: c_int = @intCast(xfm.config.hc_count);
        const hidden: c_int = @intCast(xfm.config.hidden_size);
        const width = hc * hidden;
        const w = try attn256RandBf16(rnd, &.{ hc, hidden }, s);
        defer _ = mlx.mlx_array_free(w);
        const iw = try attn256RandBf16(rnd, &.{ width, hc }, s);
        defer _ = mlx.mlx_array_free(iw);
        try testing.expect(hp.eligible(b, seq, xfm.config.hc_count, xfm.config.hidden_size, iw, .bfloat16));
        const ones = try standinOnes(&.{hidden}, s);
        defer _ = mlx.mlx_array_free(ones);
        const x = try attn256RandBf16(rnd, &.{ b, seq, hc, hidden }, s);
        defer _ = mlx.mlx_array_free(x);
        const wo = try attn256RandBf16(rnd, &.{ b, seq, 1, hidden }, s);
        defer _ = mlx.mlx_array_free(wo);
        const wi = try attn256RandBf16(rnd, &.{ b, seq, hc, 1 }, s);
        defer _ = mlx.mlx_array_free(wi);
        try testing.expect((try hp.norm(s, x, w, .{ .ctx = null }, eps, b, seq, null)) == null);
        try testing.expect((try hp.norm(s, x, w, iw, eps, b, 16, null)) == null);
        var f32_w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f32_w);
        try mlx.check(mlx.mlx_astype(&f32_w, w, .float32, s));
        try testing.expect((try hp.norm(s, x, f32_w, iw, eps, b, seq, null)) == null);
        try testing.expect((try hp.mix(s, x, x, b, seq + 1)) == null);
        const written = (try @import("../../engines/mlx/compiled.zig").applyClosure(xfm.compiled_hc_write, &.{ x, wo, wi })) orelse return error.MissingCompiledHc;
        defer _ = mlx.mlx_array_free(written);
        for ([_]bool{ false, true }) |pending| {
            const stream = if (pending) written else x;
            const n = (try hp.norm(s, x, w, iw, eps, b, seq, if (pending) .{ .out = wo, .inj = wi } else null)) orelse return error.HcFusedDeclined;
            defer n.deinit();
            var state = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(state);
            try mlx.check(mlx.mlx_reshape(&state, n.stream, &[_]c_int{ b, seq, hc, hidden }, 4, s));
            try testing.expectEqual(@as(f32, 0), try attn256MaxDiff(state, stream, s));
            var rms = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(rms);
            try mlx.check(mlx.mlx_fast_rms_norm(&rms, stream, ones, 1e-6, s));
            var normed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(normed);
            try mlx.check(mlx.mlx_multiply(&normed, rms, w, s));
            try testing.expectEqual(@as(f32, 0), try attn256MaxDiff(n.normalized, normed, s));
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            try mlx.check(mlx.mlx_reshape(&flat, normed, &[_]c_int{ b, seq, width }, 3, s));
            var raw = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(raw);
            try mlx.check(mlx.mlx_matmul(&raw, flat, iw, s));
            try testing.expectEqual(@as(f32, 0), try attn256MaxDiff(n.raw_inject, raw, s));
            const expected = (try @import("../../engines/mlx/compiled.zig").applyClosure(xfm.compiled_hc_mix, &.{ x, normed })) orelse return error.MissingCompiledHc;
            defer _ = mlx.mlx_array_free(expected);
            const mixed = (try hp.mix(s, x, normed, b, seq)) orelse return error.HcFusedDeclined;
            defer _ = mlx.mlx_array_free(mixed);
            try testing.expectEqual(@as(f32, 0), try attn256MaxDiff(mixed, expected, s));
        }
    }
}

test "hc prefill on CUDA: the fused write, norm and mix are bit-identical to the compiled chain" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const hpc = @import("../../sdk/blocks/hc/hc_prefill_cuda.zig");
    const s = mlx.gpuStream();
    var xfm = std.mem.zeroInit(Trunk, .{ .s = s, .allocator = testing.allocator });
    xfm.compileQwen4Hc();
    defer inline for (.{ xfm.compiled_hc_silu, xfm.compiled_hc_mix, xfm.compiled_hc_inj, xfm.compiled_hc_write }) |slot| {
        if (slot) |closure| _ = mlx.mlx_closure_free(closure);
    };
    try testing.expect(xfm.compiled_hc_write != null and xfm.compiled_hc_mix != null);
    var prng = std.Random.DefaultPrng.init(0x91832);
    const rnd = prng.random();
    const eps = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps);
    for ([_][2]c_int{ .{ 1, 37 }, .{ 1, 513 }, .{ 2, 65 } }) |shape| {
        const b = shape[0];
        const seq = shape[1];
        const hc: c_int = 4;
        const hidden: c_int = 2560;
        const w = try attn256RandBf16(rnd, &.{ hc, hidden }, s);
        defer _ = mlx.mlx_array_free(w);
        const ones = try standinOnes(&.{hidden}, s);
        defer _ = mlx.mlx_array_free(ones);
        const x = try attn256RandBf16(rnd, &.{ b, seq, hc, hidden }, s);
        defer _ = mlx.mlx_array_free(x);
        const wo = try attn256RandBf16(rnd, &.{ b, seq, 1, hidden }, s);
        defer _ = mlx.mlx_array_free(wo);
        const wi = try attn256RandBf16(rnd, &.{ b, seq, hc, 1 }, s);
        defer _ = mlx.mlx_array_free(wi);
        const up = try attn256RandBf16(rnd, &.{ b, seq, hc, hidden }, s);
        defer _ = mlx.mlx_array_free(up);
        const written = (try @import("../../engines/mlx/compiled.zig").applyClosure(xfm.compiled_hc_write, &.{ x, wo, wi })) orelse return error.MissingCompiledHc;
        defer _ = mlx.mlx_array_free(written);
        for ([_]bool{ false, true }) |pending| {
            const stream = if (pending) written else x;
            const n = (try hpc.norm(s, x, w, eps, b, seq, if (pending) .{ .out = wo, .inj = wi } else null)) orelse return error.HcFusedDeclined;
            defer n.deinit();
            var state = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(state);
            try mlx.check(mlx.mlx_reshape(&state, n.stream, &[_]c_int{ b, seq, hc, hidden }, 4, s));
            try testing.expect(try bf16BitEqual(state, stream, s));
            var rms = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(rms);
            try mlx.check(mlx.mlx_fast_rms_norm(&rms, stream, ones, 1e-6, s));
            var normed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(normed);
            try mlx.check(mlx.mlx_multiply(&normed, rms, w, s));
            try testing.expect(try bf16BitEqual(n.normalized, normed, s));
            const expected = (try @import("../../engines/mlx/compiled.zig").applyClosure(xfm.compiled_hc_mix, &.{ up, normed })) orelse return error.MissingCompiledHc;
            defer _ = mlx.mlx_array_free(expected);
            const mixed = (try hpc.mix(s, up, normed)) orelse return error.HcFusedDeclined;
            defer _ = mlx.mlx_array_free(mixed);
            try testing.expect(try bf16BitEqual(mixed, expected, s));
        }
    }
}

test "hc prefill: unsupported configuration and coalesced chunk bounds decline" {
    const hp = @import("../../sdk/blocks/hc/hc_prefill.zig");
    const inject = mlx.mlx_array_new_data(&[_]u16{0}, &.{1}, 1, .bfloat16);
    defer _ = mlx.mlx_array_free(inject);
    try testing.expect(hp.eligible(2, 4351, 4, 128, inject, .bfloat16));
    try testing.expect(!hp.eligible(2, 4352, 4, 128, inject, .bfloat16));
    for ([_][4]u32{ .{ 0, 17, 4, 2560 }, .{ 3, 17, 4, 2560 }, .{ 1, 16, 4, 2560 }, .{ 1, 8704, 4, 2560 }, .{ 1, 17, 0, 2560 }, .{ 1, 17, 9, 2560 }, .{ 1, 17, 4, 64 }, .{ 1, 17, 4, 4097 }, .{ 1, 17, 4, 129 } }) |shape| {
        try testing.expect(!hp.eligible(@intCast(shape[0]), @intCast(shape[1]), shape[2], shape[3], inject, .bfloat16));
    }
    try testing.expectEqual(@as(usize, @intCast(hp.max_seq)), @import("../../engines/mlx/generate.zig").nextChunkEnd(0, 8703, 8192, false, 0, 0, true, false));
}

test "hc prefill: incompatible read writes immediately" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var xfm = std.mem.zeroInit(Trunk, .{ .s = s, .allocator = testing.allocator });
    xfm.config.hc_count = 4;
    xfm.config.hidden_size = 128;
    for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dtype| {
        var h0 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h0);
        try mlx.check(mlx.mlx_ones(&h0, &.{ 1, 17, 512 }, 3, dtype, s));
        var out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_ones(&out, &.{ 1, 17, 128 }, 3, dtype, s));
        var inj = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inj);
        try mlx.check(mlx.mlx_ones(&inj, &.{ 1, 17, 4, 1 }, 4, dtype, s));
        for ([_]?mlx.mlx_dtype{ null, .uint32, .float32, .bfloat16 }) |inject_dtype| {
            var iw = mlx.mlx_array{ .ctx = null };
            defer if (iw.ctx != null) {
                _ = mlx.mlx_array_free(iw);
            };
            if (inject_dtype) |dt| {
                iw = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_ones(&iw, &.{ 512, 4 }, 2, dt, s));
            }
            var h = try standinRef(h0);
            defer _ = mlx.mlx_array_free(h);
            var pending: ?HcPending = null;
            defer if (pending) |*p| p.deinit();
            try xfm.hcWriteOrDefer(&h, out, inj, 1, 17, iw, &pending);
            const can_defer = dtype == .bfloat16 and inject_dtype == .bfloat16;
            try testing.expectEqual(can_defer, pending != null);
            try xfm.hcFlush(&h, 1, 17, &pending);
            const expected = try xfm.hcWrite(try standinRef(h0), out, inj, 1, 17);
            defer _ = mlx.mlx_array_free(expected);
            try testing.expectEqual(@as(f32, 0), try attn256MaxDiff(h, expected, s));
        }
    }
}
